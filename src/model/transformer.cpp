#include "transformer.hpp"
#include "model.hpp"
#include <cmath>
#include <iostream>
#include <fstream> 
#include <fcntl.h>
#include <sys/mman.h>


void CTransformer::mapWeightsToMemory(CModelConfig* modelConfig, float* ptr, int sharedWeights){
    const int headSize = modelConfig->dim / modelConfig->numHeads;
    const uint64_t numLayers = modelConfig->numLayers;
    float* currentPtr = ptr; 


    w.tokenEmbeddingTable = currentPtr;
    currentPtr += modelConfig->vocabSize * modelConfig->dim;


    w.rmsAttWeight = currentPtr;
    currentPtr += numLayers * modelConfig->dim;
    
    w.wq = currentPtr;
    currentPtr += numLayers * modelConfig->dim * (modelConfig->numHeads * headSize);
    
    w.wk = currentPtr;
    currentPtr += numLayers * modelConfig->dim * (modelConfig->numKvHeads * headSize);
    
    w.wv = currentPtr;
    currentPtr += numLayers * modelConfig->dim * (modelConfig->numKvHeads * headSize);
    
    w.wo = currentPtr;
    currentPtr += numLayers * (modelConfig->numHeads * headSize) * modelConfig->dim;

    w.rmsFfnWeight = currentPtr;
    currentPtr += numLayers * modelConfig->dim;
    
    w.w1 = currentPtr;
    currentPtr += numLayers * modelConfig->dim * modelConfig->feedForwardDim;
    
    w.w2 = currentPtr;
    currentPtr += numLayers * modelConfig->feedForwardDim * modelConfig->dim;
    
    w.w3 = currentPtr;
    currentPtr += numLayers * modelConfig->dim * modelConfig->feedForwardDim;


    w.rmsFinalWeight = currentPtr;
    currentPtr += modelConfig->dim;


    currentPtr += modelConfig->maxSeqLen * headSize / 2; 
    currentPtr += modelConfig->maxSeqLen * headSize / 2; 


    w.wcls = sharedWeights ? w.tokenEmbeddingTable : currentPtr;
}


void CTransformer::load(const std::string& checkpointPath, CModelConfig* modelConfig, int* fileDescriptor, float** data, ssize_t* totalFileSize) {
    

    std::ifstream fileStream(checkpointPath, std::ios::binary | std::ios::ate);
    if (!fileStream) {
        std::cerr<<"[ERROR:] Unable to open checkpoint file: " << checkpointPath<<std::endl;
    }


    *totalFileSize = fileStream.tellg();
    fileStream.seekg(0, std::ios::beg);


    fileStream.read(reinterpret_cast<char*>(modelConfig), sizeof(CModelConfig));
    if (!fileStream) {
         std::cerr<<"[ERROR:] Unable to read model config"<<std::endl;
    }

    bool sharedWeights = modelConfig->vocabSize > 0;
    modelConfig->vocabSize = std::abs(modelConfig->vocabSize);

    fileStream.close();


    *fileDescriptor = open(checkpointPath.c_str(), O_RDONLY);
    if (*fileDescriptor == -1) {
        std::cerr<<"[ERROR:] Unable to open file descriptor for: " << checkpointPath<<std::endl;
    }


    *data = static_cast<float*>(mmap(nullptr, *totalFileSize, PROT_READ, MAP_PRIVATE, *fileDescriptor, 0));
    if (*data == MAP_FAILED) {
        std::cerr<<"[ERROR:] Unable to memory-map the file:  " << checkpointPath<<std::endl;
    }

    constexpr uint64_t configSizeInFloats = sizeof(CModelConfig) / sizeof(float);
    float* weightsPtr = *data + configSizeInFloats;


    mapWeightsToMemory(modelConfig, weightsPtr, sharedWeights);
}


void CTransformer::initializeModel(const std::string checkpointPath) {
    load(checkpointPath, &config, &fd, &data, &fileSize);
    state.allocateMemory(&config);
}


void CTransformer::freeModel() {

    if (data != MAP_FAILED) {
        munmap(data, fileSize);
    }
    
    if (fd != -1) {
        close(fd);
    }
    
    state.deallocateMemory();
}


float* CTransformer::forward(int token, int pos, CBackend *backend) {
    CModelConfig* config = &this->config;
    CRunState* state = &this->state;
    

    float* inputVec = state->currentActivation;
    const int embeddingDim = config->dim;
    const int kvDim = (config->dim * config->numKvHeads) / config->numHeads;
    const int kvHeadMultiplier = config->numHeads / config->numKvHeads;
    const int headSize = embeddingDim / config->numHeads;
    const int ffnHiddenDim = config->feedForwardDim;

    float* tokenEmbedding = w.tokenEmbeddingTable + token * embeddingDim;
    std::memcpy(inputVec, tokenEmbedding, embeddingDim * sizeof(float));

    for (uint64_t layer = 0; layer < config->numLayers; ++layer) {

        backend->rmsnorm(state->branchActivation, inputVec, w.rmsAttWeight + layer * embeddingDim, embeddingDim);


        const int kvCacheOffset = layer * config->maxSeqLen * kvDim;
        state->k = state->keyCache + kvCacheOffset + pos * kvDim;
        state->v = state->valueCache + kvCacheOffset + pos * kvDim;
        

        backend->matmul(state->q, state->branchActivation, w.wq + layer * embeddingDim * embeddingDim, embeddingDim, embeddingDim);
        backend->matmul(state->k, state->branchActivation, w.wk + layer * embeddingDim * kvDim, embeddingDim, kvDim);
        backend->matmul(state->v, state->branchActivation, w.wv + layer * embeddingDim * kvDim, embeddingDim, kvDim);
        backend->ropeEncoding(state->q, state->k, headSize, pos, embeddingDim, kvDim);
        for (int headIdx = 0; headIdx < config->numHeads; ++headIdx) {
            float* query = state->q + headIdx * headSize;
            float* attentionScores = state->attentionScores + headIdx * config->maxSeqLen;
            
            backend->gemvQkSeq(query,state->keyCache+kvCacheOffset+(headIdx / kvHeadMultiplier) * headSize,attentionScores, pos, kvDim, headSize);

            backend->softmax(attentionScores, pos + 1);

            float* headOutput = state->branchActivation + headIdx * headSize;
            std::memset(headOutput, 0, headSize * sizeof(float));

            backend->weightedV(headOutput, state->valueCache + kvCacheOffset + (headIdx / kvHeadMultiplier) * headSize, attentionScores, pos, kvDim, headSize);
        }

        backend->matmul(state->extraBuffer, state->branchActivation, w.wo + layer * embeddingDim * embeddingDim, embeddingDim, embeddingDim);
        
        backend->axpy(inputVec, state->extraBuffer, 1.f, embeddingDim);


        backend->rmsnorm(state->branchActivation, inputVec, w.rmsFfnWeight + layer * embeddingDim, embeddingDim);
        

        backend->matmul(state->hiddenBuffer, state->branchActivation, w.w1 + layer * embeddingDim * ffnHiddenDim, embeddingDim, ffnHiddenDim);
        backend->matmul(state->extraHiddenBuffer, state->branchActivation, w.w3 + layer * embeddingDim * ffnHiddenDim, embeddingDim, ffnHiddenDim);
        
        
        for (int i = 0; i < ffnHiddenDim; ++i) {
            const float sigmoid = 1.0f / (1.0f + std::exp(-state->hiddenBuffer[i]));
            state->hiddenBuffer[i] = state->hiddenBuffer[i] * sigmoid * state->extraHiddenBuffer[i];
        }

        backend->matmul(state->branchActivation, state->hiddenBuffer, w.w2 + layer * ffnHiddenDim * embeddingDim, ffnHiddenDim, embeddingDim);
        backend->axpy(inputVec, state->branchActivation, 1.f, embeddingDim);

    }

    backend->rmsnorm(inputVec, inputVec, w.rmsFinalWeight, embeddingDim);
    backend->matmul(state->logits, inputVec, w.wcls, embeddingDim, config->vocabSize);
    
    return state->logits;
}

