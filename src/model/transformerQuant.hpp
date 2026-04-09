#ifndef CTRANSFORMERQUANT_HPP
#define CTRANSFORMERQUANT_HPP

#include "model.hpp"
#include "modelConfig.hpp"
#include "../backend/backendQuant.hpp"
#include "../infer/runState.hpp"
#include <unistd.h>
#include <string>

class CTransformerQuant:public CModel{
public:
    
    float* scale = new float;
    void load(const std::string& checkpointPath, CModelConfig* modelConfig, int* fileDescriptor, float** data, ssize_t* totalFileSize);
    float* forward(int token, int pos, CBackend *backend);
    void mapWeightsToMemoryQuant(CModelConfig* modelConfig, float* ptr, int sharedWeights);
    void exportQuantizedModel(CModelConfig* modelConfig, float* ptr, int sharedWeights, const std::string& outputFile);

};

#endif