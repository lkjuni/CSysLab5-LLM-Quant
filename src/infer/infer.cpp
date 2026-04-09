#include <string>
#include <cstring>
#include <iostream>
#include <chrono>
#include "infer.hpp"
#include "../model/transformer.hpp"
#include "../util.hpp"

CInfer::CInfer() {
    this->mt = MODEL_LLAMA;
    this->bt = CPU;
    this->model = NULL;
    this->backend = NULL;

    this->maxSeqLen = 256;
    this->temperature = 0.0;    // 0.0：贪婪解码；请勿设置超过1.0
    this->topp = 1.0f;          // 核采样中的top-p值。1.0：关闭。推荐设置为0.9
    this->steps = 256;          // 运行的步骤数
    this->rngSeed = 0;          // 随机数种子
}

CInfer::~CInfer(){
    if(this->model != NULL){
        delete this->model;
    }

    if(this->backend != NULL){
        delete this->backend;
    }
}

long timeInMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

void printSafeString(std::string_view piece) {

    if (piece.empty()) {
        return;
    }
    if (piece.size() == 1) {
        unsigned char byteVal = static_cast<unsigned char>(piece[0]);
        if (!(std::isprint(byteVal) || std::isspace(byteVal))) {
            return; 
        }
    }
    std::cout << piece;
}

void CInfer::build(std::string modelPath, std::string tknzrPath, ModelType mt, BackendType bt, bool enableQuantization,bool exportQuantizedModel,std::string exportFilePath){
    if(mt == MODEL_LLAMA){
       if(!enableQuantization){
           model = new CTransformer();   
       }
       else{
           model = new CTransformerQuant();   
           if(exportQuantizedModel){
               model->mode = 0;
               model->outputFile = exportFilePath;
           }
       }
       model->initializeModel(modelPath);
    }else if(mt == MODEL_DEEPSEEK){
       std::cerr<< "[ERROR:] Unsupported model type\n"<<std::endl;
       exit(1);
    }else{
       std::cerr<< "[ERROR:] Unsupported model type\n"<<std::endl;
       exit(1);
   }
     if(bt == CPU){
       if(!enableQuantization){
           backend = new CBackend();
       }
       else{
           backend = new CBackendQuant();
       }
     }else if(bt == CPU_X86){
       backend = new CBackend();
     }else if(bt == CPU_ARM){
       backend = new CBackend();
     }else{
       std::cerr<< "[ERROR:] Unsupported backend type\n"<<std::endl;
     }
   tokenizer = new CTokenizer();
   sampler = new CSampler();
   
   tokenizer->initializeTokenizer(tknzrPath,model->config.vocabSize);

   sampler->initializeSampler(model->config.vocabSize, temperature, topp, rngSeed);
  
}



void CInfer::generate(std::string prompt) {

    std::string emptyPrompt = "";

    int numPromptTokens = 0;
    int* promptTokens = new int[prompt.size() + 3]; // BOS, EOS, 和空终止符

    model->encode(tokenizer, prompt, 1, 0, promptTokens, &numPromptTokens);

    if (numPromptTokens < 1) {
        std::cerr<<"[ERROR:] Something is wrong, expected at least 1 prompt token\n"<<std::endl;
        exit(EXIT_FAILURE);
    }

    printf("[MSG:] The length of the prompt -> %d\n[MSG:] Prompt tokens: ", numPromptTokens);
    for (int i = 0; i < numPromptTokens; i++) {
        printf("%d ", promptTokens[i]);
    }
    printf("\n");

    long start = 0;              
    int next;                    
    int token = promptTokens[0]; 
    int pos = 0;               

    while (pos < steps) {

        float* logits = model->forward(token, pos, backend);

        if (pos < numPromptTokens - 1) {

            next = promptTokens[pos + 1];

        } else {
            next = sampler->sample(logits, backend);
        }
        pos++;

        if (next == 1) { 
            break; 
        }

        char* decodedToken = model->decode(tokenizer, token, next);
        printSafeString(decodedToken);
        fflush(stdout);
        token = next;

        if (start == 0) { 
            start = timeInMs(); 
        }
    }
    printf("\n");

    if (pos > 1) {
        long end = timeInMs();
        fprintf(stderr, "Achieved token/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }

    delete[] promptTokens;
}
