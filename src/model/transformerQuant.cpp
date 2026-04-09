#include "transformerQuant.hpp"
#include "model.hpp"
#include <cmath>
#include <iostream>
#include <fstream> 
#include <fcntl.h>
#include <sys/mman.h>
#include <iomanip> 
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring> 
#include <cfloat>


void computeGlobalScale(float* start1, float* end1, float* start2, float* end2, float& scale) {
    /**    
     *TODO: 计算两个连续内存块中的全局缩放因子, 计算两块内存中所有元素的最大绝对值max，返回缩放因子scale=max/127.0
     * 
     * @param[in]  start1 指向第一个连续内存块起始位置的指针
     * @param[in]  end1 指向第一个连续内存块结束位置的指针（不包含该位置）
     * @param[in]  start2 指向第二个连续内存块起始位置的指针
     * @param[in]  end2 指向第二个连续内存块结束位置的指针（不包含该位置）
     * @param[out] scale 计算得到的缩放因子，存储在该变量中
     *
     **/
    }
void quantizeToInt8(float* src, int8_t* dst, size_t size, float scale) {
     /**    
     *TODO: 将浮点数数组量化为 int8_t 数组
     * 
     * @param[in]  src 指向输入数据（浮点数数组）的指针，表示要量化的原始数据所在的内存空间
     * @param[out] dst 指向输出数据（int8 数组）的指针，量化后的数据会存储到这个内存空间
     * @param[in]  size 数组大小，src 和 dst 中的元素数量
     * @param[in]  scale 量化缩放因子，用于将浮点数转换为 int8_t，由 computeGlobalScale计算得到
     * 
     **/
    }
    

void writeDataToFile(std::ofstream& outFile, float* basePtr, float* currentPtr) {
    size_t totalSize = (currentPtr - basePtr) * sizeof(float);
    outFile.write(reinterpret_cast<char*>(basePtr), totalSize);
    outFile.flush();
}
void quantizeAndWrite(float* startPtr, float* endPtr, 
                      std::ofstream& outFile, float globalScale) {
    size_t size = endPtr - startPtr; 
    int8_t* quantizedData = new int8_t[size];

    quantizeToInt8(startPtr, quantizedData, size, globalScale);

    outFile.write(reinterpret_cast<char*>(quantizedData), size * sizeof(int8_t));
    outFile.flush();

    delete[] quantizedData;
}
void CTransformerQuant::exportQuantizedModel(CModelConfig* modelConfig, float* ptr, int sharedWeights, const std::string& outputFile) {
    const int headSize = modelConfig->dim / modelConfig->numHeads;
    const uint64_t numLayers = modelConfig->numLayers;
    float* currentPtr = ptr;

    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile) {
        std::cerr << "[ERROR:] Cannot open file " << outputFile << " for writing." << std::endl;
        return;
    }

    outFile.write(reinterpret_cast<const char*>(&config.dim), sizeof(config.dim));
    outFile.write(reinterpret_cast<const char*>(&config.feedForwardDim), sizeof(config.feedForwardDim));
    outFile.write(reinterpret_cast<const char*>(&config.numLayers), sizeof(config.numLayers));
    outFile.write(reinterpret_cast<const char*>(&config.numHeads), sizeof(config.numHeads));
    outFile.write(reinterpret_cast<const char*>(&config.numKvHeads), sizeof(config.numKvHeads));
    outFile.write(reinterpret_cast<const char*>(&config.vocabSize), sizeof(config.vocabSize));
    outFile.write(reinterpret_cast<const char*>(&config.maxSeqLen), sizeof(config.maxSeqLen));

    float* basePtr = currentPtr;
    currentPtr += modelConfig->vocabSize * modelConfig->dim;
    
    writeDataToFile(outFile, basePtr, currentPtr);
    basePtr = currentPtr;

    currentPtr += numLayers * modelConfig->dim;
    writeDataToFile(outFile, basePtr, currentPtr);

    float* qkvPtrStar = currentPtr;
    currentPtr += numLayers * modelConfig->dim * (modelConfig->numHeads * headSize);

    currentPtr += numLayers * modelConfig->dim * (modelConfig->numKvHeads * headSize);

    currentPtr += numLayers * modelConfig->dim * (modelConfig->numKvHeads * headSize);

    currentPtr += numLayers * (modelConfig->numHeads * headSize) * modelConfig->dim;
    
    float* qkvPtrEnd = currentPtr;

    basePtr = currentPtr ;
    currentPtr += numLayers * modelConfig->dim;

    writeDataToFile(outFile, basePtr, currentPtr);

    float* ffnPtrStart = currentPtr;
    currentPtr += numLayers * modelConfig->dim * modelConfig->feedForwardDim;

    currentPtr += numLayers * modelConfig->feedForwardDim * modelConfig->dim;

    currentPtr += numLayers * modelConfig->dim * modelConfig->feedForwardDim;

    float* ffnPtrEnd = currentPtr;

    basePtr = currentPtr;
    currentPtr += modelConfig->dim;
    writeDataToFile(outFile, basePtr, currentPtr);

    computeGlobalScale(qkvPtrStar, qkvPtrEnd, ffnPtrStart, ffnPtrEnd, *scale);

    quantizeAndWrite(qkvPtrStar, qkvPtrEnd,outFile, *scale);
    quantizeAndWrite(ffnPtrStart, ffnPtrEnd,outFile, *scale);
    
    outFile.write(reinterpret_cast<char*>(scale), sizeof(float));

    outFile.close();
    std::cout << "[MSG:] Quantized model saved to " << outputFile << std::endl;
    exit(1);
}
void CTransformerQuant::mapWeightsToMemoryQuant(CModelConfig* modelConfig, float* ptr, int sharedWeights){
    const int headSize = modelConfig->dim / modelConfig->numHeads;
    const uint64_t numLayers = modelConfig->numLayers;
    float* currentPtr = ptr; 

    w.tokenEmbeddingTable = currentPtr;
    currentPtr += modelConfig->vocabSize * modelConfig->dim;
    w.rmsAttWeight = currentPtr;
    currentPtr += numLayers * modelConfig->dim;
    w.rmsFfnWeight = currentPtr;
    currentPtr += numLayers * modelConfig->dim;
    w.rmsFinalWeight = currentPtr;
    currentPtr += modelConfig->dim;

    w.wq = currentPtr;
    currentPtr += (numLayers * modelConfig->dim * (modelConfig->numHeads * headSize))/4;
    
    w.wk = currentPtr;
    currentPtr += (numLayers * modelConfig->dim * (modelConfig->numKvHeads * headSize))/4;
    
    w.wv = currentPtr;
    currentPtr += (numLayers * modelConfig->dim * (modelConfig->numKvHeads * headSize))/4;
    
    w.wo = currentPtr;
    currentPtr += (numLayers * (modelConfig->numHeads * headSize) * modelConfig->dim)/4;

    w.w1 = currentPtr;
    currentPtr += (numLayers * modelConfig->dim * modelConfig->feedForwardDim)/4;
    
    w.w2 = currentPtr;
    currentPtr += (numLayers * modelConfig->feedForwardDim * modelConfig->dim)/4;
    
    w.w3 = currentPtr;
    currentPtr += (numLayers * modelConfig->dim * modelConfig->feedForwardDim)/4;

    scale = currentPtr;
    w.wcls = sharedWeights ? w.tokenEmbeddingTable : currentPtr;
}

void CTransformerQuant::load(const std::string& checkpointPath, CModelConfig* modelConfig, int* fileDescriptor, float** data, ssize_t* totalFileSize) {
    
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
    if (mode == 1){
        mapWeightsToMemoryQuant(modelConfig, weightsPtr, sharedWeights);
    }
    else if(mode == 0){
        exportQuantizedModel(modelConfig, weightsPtr, sharedWeights,outputFile);
    }
    
}

float* CTransformerQuant::forward(int token, int pos, CBackend *cbackend) {
    CBackendQuant* backend = dynamic_cast<CBackendQuant*>(cbackend);
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
          
        backend->matmulQuant(state->q, state->branchActivation, w.wq + (layer * embeddingDim * embeddingDim)/4, scale,embeddingDim, embeddingDim);
        backend->matmulQuant(state->k, state->branchActivation, w.wk + (layer * embeddingDim * kvDim)/4, scale, embeddingDim, kvDim);
        backend->matmulQuant(state->v, state->branchActivation, w.wv + (layer * embeddingDim * kvDim)/4, scale, embeddingDim, kvDim);
        
        backend->ropeEncoding(state->q, state->k, headSize, pos, embeddingDim, kvDim);
        #pragma omp parallel for
        for (int headIdx = 0; headIdx < config->numHeads; ++headIdx) {
            float* query = state->q + headIdx * headSize;
            float* attentionScores = state->attentionScores + headIdx * config->maxSeqLen;
            
            backend->gemvQkSeq(query,state->keyCache+kvCacheOffset+(headIdx / kvHeadMultiplier) * headSize,attentionScores, pos, kvDim, headSize);

            backend->softmax(attentionScores, pos + 1);

            float* headOutput = state->branchActivation + headIdx * headSize;
            std::memset(headOutput, 0, headSize * sizeof(float));

            backend->weightedV(headOutput, state->valueCache + kvCacheOffset + (headIdx / kvHeadMultiplier) * headSize, attentionScores, pos, kvDim, headSize);
        }

        backend->matmulQuant(state->extraBuffer, state->branchActivation, w.wo + (layer * embeddingDim * embeddingDim)/4, scale,embeddingDim, embeddingDim);
        
        backend->axpy(inputVec, state->extraBuffer, 1.f, embeddingDim);

        backend->rmsnorm(state->branchActivation, inputVec, w.rmsFfnWeight + layer * embeddingDim, embeddingDim);
        
        backend->matmulQuant(state->hiddenBuffer, state->branchActivation, w.w1 + (layer * embeddingDim * ffnHiddenDim)/4, scale, embeddingDim, ffnHiddenDim);
        backend->matmulQuant(state->extraHiddenBuffer, state->branchActivation, w.w3 + (layer * embeddingDim * ffnHiddenDim)/4, scale, embeddingDim, ffnHiddenDim);
        
        for (int i = 0; i < ffnHiddenDim; ++i) {
            const float sigmoid = 1.0f / (1.0f + std::exp(-state->hiddenBuffer[i]));
            state->hiddenBuffer[i] = state->hiddenBuffer[i] * sigmoid * state->extraHiddenBuffer[i];
        }

        backend->matmulQuant(state->branchActivation, state->hiddenBuffer, w.w2 + (layer * ffnHiddenDim * embeddingDim)/4, scale, ffnHiddenDim, embeddingDim);
        
       
        backend->axpy(inputVec, state->branchActivation, 1.f, embeddingDim);

    }

    backend->rmsnorm(inputVec, inputVec, w.rmsFinalWeight, embeddingDim);
    backend->matmul(state->logits, inputVec, w.wcls, embeddingDim, config->vocabSize);
    
    return state->logits;
}


