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
    // 计算两个连续内存块中所有元素的最大绝对值
    float max_abs = 0.0f;

    // 遍历第一个内存块
    for (float* ptr = start1; ptr < end1; ptr++) {
        float abs_val = std::fabs(*ptr);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }

    // 遍历第二个内存块
    for (float* ptr = start2; ptr < end2; ptr++) {
        float abs_val = std::fabs(*ptr);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }

    // 计算缩放因子：scale = max_abs / 127.0
    // 这样最大值会被映射到 int8 的 ±127 范围
    scale = max_abs / 127.0f;
}
void quantizeToInt8(float* src, int8_t* dst, size_t size, float scale) {
    // 将浮点数数组量化为 int8_t 数组
    // 量化公式：dst[i] = (int8_t)round(src[i] / scale)
    // 值会被限制在 [-128, 127] 范围内

    for (size_t i = 0; i < size; i++) {
        // 计算量化值
        float quantized_val = src[i] / scale;

        // 四舍五入
        int32_t rounded_val = (int32_t)std::round(quantized_val);

        // 限制在 int8 范围内 [-128, 127]
        if (rounded_val > 127) {
            rounded_val = 127;
        } else if (rounded_val < -128) {
            rounded_val = -128;
        }

        dst[i] = (int8_t)rounded_val;
    }
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

    // 写入模型配置
    outFile.write(reinterpret_cast<const char*>(&config.dim), sizeof(config.dim));
    outFile.write(reinterpret_cast<const char*>(&config.feedForwardDim), sizeof(config.feedForwardDim));
    outFile.write(reinterpret_cast<const char*>(&config.numLayers), sizeof(config.numLayers));
    outFile.write(reinterpret_cast<const char*>(&config.numHeads), sizeof(config.numHeads));
    outFile.write(reinterpret_cast<const char*>(&config.numKvHeads), sizeof(config.numKvHeads));
    outFile.write(reinterpret_cast<const char*>(&config.vocabSize), sizeof(config.vocabSize));
    outFile.write(reinterpret_cast<const char*>(&config.maxSeqLen), sizeof(config.maxSeqLen));

    // 写入非量化的权重（token embedding 和 norm weights）
    float* basePtr = currentPtr;
    currentPtr += modelConfig->vocabSize * modelConfig->dim;
    writeDataToFile(outFile, basePtr, currentPtr);

    basePtr = currentPtr;
    currentPtr += numLayers * modelConfig->dim;
    writeDataToFile(outFile, basePtr, currentPtr);

    // 标记需要量化的权重
    float* wqPtr = currentPtr;
    currentPtr += numLayers * modelConfig->dim * (modelConfig->numHeads * headSize);

    float* wkPtr = currentPtr;
    currentPtr += numLayers * modelConfig->dim * (modelConfig->numKvHeads * headSize);

    float* wvPtr = currentPtr;
    currentPtr += numLayers * modelConfig->dim * (modelConfig->numKvHeads * headSize);

    float* woPtr = currentPtr;
    currentPtr += numLayers * (modelConfig->numHeads * headSize) * modelConfig->dim;

    basePtr = currentPtr;
    currentPtr += numLayers * modelConfig->dim;
    writeDataToFile(outFile, basePtr, currentPtr);

    float* w1Ptr = currentPtr;
    currentPtr += numLayers * modelConfig->dim * modelConfig->feedForwardDim;

    float* w2Ptr = currentPtr;
    currentPtr += numLayers * modelConfig->feedForwardDim * modelConfig->dim;

    float* w3Ptr = currentPtr;
    currentPtr += numLayers * modelConfig->dim * modelConfig->feedForwardDim;

    basePtr = currentPtr;
    currentPtr += modelConfig->dim;
    writeDataToFile(outFile, basePtr, currentPtr);

    // 分别计算每个权重矩阵的局部缩放因子
    size_t wqSize = numLayers * modelConfig->dim * (modelConfig->numHeads * headSize);
    size_t wkSize = numLayers * modelConfig->dim * (modelConfig->numKvHeads * headSize);
    size_t wvSize = numLayers * modelConfig->dim * (modelConfig->numKvHeads * headSize);
    size_t woSize = numLayers * (modelConfig->numHeads * headSize) * modelConfig->dim;
    size_t w1Size = numLayers * modelConfig->dim * modelConfig->feedForwardDim;
    size_t w2Size = numLayers * modelConfig->feedForwardDim * modelConfig->dim;
    size_t w3Size = numLayers * modelConfig->dim * modelConfig->feedForwardDim;

    // 计算最大绝对值作为缩放因子 (for each weight matrix)
    auto computeMaxAbs = [](float* data, size_t size) -> float {
        float max_abs = 0.0f;
        for (size_t i = 0; i < size; i++) {
            float abs_val = std::fabs(data[i]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }
        return max_abs / 127.0f;
    };

    // 为每个权重矩阵计算缩放因子 (indices: 0=wq, 1=wk, 2=wv, 3=wo, 4=w1, 5=w2, 6=w3)
    scales[0] = computeMaxAbs(wqPtr, wqSize);   // wq
    scales[1] = computeMaxAbs(wkPtr, wkSize);   // wk
    scales[2] = computeMaxAbs(wvPtr, wvSize);   // wv
    scales[3] = computeMaxAbs(woPtr, woSize);   // wo
    scales[4] = computeMaxAbs(w1Ptr, w1Size);   // w1
    scales[5] = computeMaxAbs(w2Ptr, w2Size);   // w2
    scales[6] = computeMaxAbs(w3Ptr, w3Size);   // w3

    // 使用各自的缩放因子对权重进行量化并写入
    quantizeAndWrite(wqPtr, wqPtr + wqSize, outFile, scales[0]);
    quantizeAndWrite(wkPtr, wkPtr + wkSize, outFile, scales[1]);
    quantizeAndWrite(wvPtr, wvPtr + wvSize, outFile, scales[2]);
    quantizeAndWrite(woPtr, woPtr + woSize, outFile, scales[3]);
    quantizeAndWrite(w1Ptr, w1Ptr + w1Size, outFile, scales[4]);
    quantizeAndWrite(w2Ptr, w2Ptr + w2Size, outFile, scales[5]);
    quantizeAndWrite(w3Ptr, w3Ptr + w3Size, outFile, scales[6]);

    // 写入7个缩放因子
    outFile.write(reinterpret_cast<char*>(scales), 7 * sizeof(float));

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

    // 加载7个局部缩放因子 [wq, wk, wv, wo, w1, w2, w3]
    scales = (float*)currentPtr;
    currentPtr += 7;

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

        // 使用各自的局部缩放因子
        backend->matmulQuant(state->q, state->branchActivation, w.wq + (layer * embeddingDim * embeddingDim)/4, scales + 0, embeddingDim, embeddingDim);  // wq scale
        backend->matmulQuant(state->k, state->branchActivation, w.wk + (layer * embeddingDim * kvDim)/4, scales + 1, embeddingDim, kvDim);  // wk scale
        backend->matmulQuant(state->v, state->branchActivation, w.wv + (layer * embeddingDim * kvDim)/4, scales + 2, embeddingDim, kvDim);  // wv scale

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

        backend->matmulQuant(state->extraBuffer, state->branchActivation, w.wo + (layer * embeddingDim * embeddingDim)/4, scales + 3, embeddingDim, embeddingDim);  // wo scale

        backend->axpy(inputVec, state->extraBuffer, 1.f, embeddingDim);

        backend->rmsnorm(state->branchActivation, inputVec, w.rmsFfnWeight + layer * embeddingDim, embeddingDim);

        backend->matmulQuant(state->hiddenBuffer, state->branchActivation, w.w1 + (layer * embeddingDim * ffnHiddenDim)/4, scales + 4, embeddingDim, ffnHiddenDim);  // w1 scale
        backend->matmulQuant(state->extraHiddenBuffer, state->branchActivation, w.w3 + (layer * embeddingDim * ffnHiddenDim)/4, scales + 6, embeddingDim, ffnHiddenDim);  // w3 scale

        for (int i = 0; i < ffnHiddenDim; ++i) {
            const float sigmoid = 1.0f / (1.0f + std::exp(-state->hiddenBuffer[i]));
            state->hiddenBuffer[i] = state->hiddenBuffer[i] * sigmoid * state->extraHiddenBuffer[i];
        }

        backend->matmulQuant(state->branchActivation, state->hiddenBuffer, w.w2 + (layer * ffnHiddenDim * embeddingDim)/4, scales + 5, ffnHiddenDim, embeddingDim);  // w2 scale


        backend->axpy(inputVec, state->branchActivation, 1.f, embeddingDim);

    }

    backend->rmsnorm(inputVec, inputVec, w.rmsFinalWeight, embeddingDim);
    backend->matmul(state->logits, inputVec, w.wcls, embeddingDim, config->vocabSize);

    return state->logits;
}


