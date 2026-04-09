#ifndef INFER_HPP
#define INFER_HPP


#include "../model/model.hpp"
#include "../model/transformer.hpp"
#include "../model/transformerQuant.hpp"
#include "../backend/backend.hpp"
#include "../backend/backendQuant.hpp"
#include "../util.hpp"
#include "sampler.hpp"

#include<string>

class CInfer{
    private:
        enum ModelType mt;
        enum BackendType bt;

        CBackend *backend;
        CModel *model;
        CTokenizer *tokenizer;
        CSampler *sampler;

        int maxSeqLen;
        float temperature;   
        float topp;          
        int steps;
        unsigned long long rngSeed;

    public:
        CInfer();
        ~CInfer();
        
        void build(std::string modelPath, std::string tknzrPath, ModelType mt, BackendType bt, bool enableQuantization,bool exportQuantizedModel,std::string exportFilePath);
        void generate(std::string prompt);
};

#endif