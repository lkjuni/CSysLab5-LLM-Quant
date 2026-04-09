#ifndef CTRANSFORMER_HPP
#define CTRANSFORMER_HPP

#include "model.hpp"
#include "modelConfig.hpp"
#include "../infer/runState.hpp"
#include <unistd.h>
#include <string>

class CTransformer:public CModel{
public:

    void load(const std::string& checkpointPath, CModelConfig* modelConfig, int* fileDescriptor, float** data, ssize_t* totalFileSize);
    float* forward(int token, int pos, CBackend *backend);
    void mapWeightsToMemory(CModelConfig* p, float* ptr, int sharedWeights);
    void initializeModel(const std::string modelPath);
    void freeModel();
};

#endif