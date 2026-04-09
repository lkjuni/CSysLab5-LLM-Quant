#include <iostream>
#include <string>
#include <map>
#include "./infer/infer.hpp"
#include "util.hpp"

std::string modelPath;
std::string tknzrPath;
std::string prompt;  
std::string exportFilePath;
ModelType mt;
BackendType bt;
bool openKvCache = 1;
bool enableQuantization = false;
bool exportQuantizedModel = false;  
void config(){

}
void parse(int argc, char* argv[]) {
    if (argc <= 1 || argv[1] == nullptr || std::string(argv[1]).empty() ||
        argc <= 2 || argv[2] == nullptr || std::string(argv[2]).empty()) {
        std::cout << "[MSG:] Usage: muduo [model_path] [tokenizer_path] "
        "--prompt <prompt> --modelType <llama> --backend <cpu|cpu_x86|cpu_arm|cuda>  --quantize <export|load|0>" << std::endl;
        exit(1);
    }

    modelPath = argv[1];
    tknzrPath = argv[2];

    std::map<std::string, std::string> args;
    for (int i = 3; i < argc - 1; i += 2) {
        std::string key = argv[i];
        std::string value = argv[i + 1];
        args[key] = value;
    }

    if (args.count("--prompt")) {
        prompt = args["--prompt"];
    } else {
        std::cout << "[MSG:] Using default prompt \"once upon a time,\"\n";
    }

    if (args.count("--modelType")) {
        std::string modelTypeStr = args["--modelType"];
        if (modelTypeStr == "llama") {
            mt = ModelType::MODEL_LLAMA;
        } else if (modelTypeStr == "deepseek") {
            mt = ModelType::MODEL_DEEPSEEK;
        } else {
            std::cerr << "Unknown model type: " << modelTypeStr << std::endl;
        }
    }

    if (args.count("--backend")) {
        std::string backendStr = args["--backend"];
        if (backendStr == "cpu") {
            bt = BackendType::CPU;
        } else if (backendStr == "cpu_x86") {
            bt = BackendType::CPU_X86;
        } else if (backendStr == "cpu_arm") {
            bt = BackendType::CPU_ARM;
        } else if (backendStr == "cuda") {
            bt = BackendType::CUDA;
        } else {
            std::cerr << "[ERROR:] Unsupported backend: " << backendStr << std::endl;
            exit(1);
        }
    }


    if (args.count("--quantize")) {
        std::string quantizeFlag = args["--quantize"];
        if (quantizeFlag == "export") {
            enableQuantization = true;
            exportQuantizedModel = true;  
            std::cout << "[MSG:] Exporting quantized model." << std::endl;
            if (argc >= 6 && argv[argc - 1] != nullptr) {  
                exportFilePath = argv[argc - 1];  
                std::cout << "[MSG:] Exporting quantized model to: " << exportFilePath << std::endl;
            } else {
                std::cerr << "[ERROR:] Missing export path for --quantize e mode." << std::endl;
                exit(1);
            }
        } else if (quantizeFlag == "load") {
            enableQuantization = true;
            exportQuantizedModel = false;  
            std::cout << "[MSG:] Quantization enabled for inference." << std::endl;
        } else if (quantizeFlag != "0") {
            std::cerr << "[ERROR:] Invalid value for --quantize. Use 'e', '1', or '0'." << std::endl;
            exit(1);
        }
    }
}

void init(){
    modelPath = "";        
    tknzrPath = "";    
    exportFilePath = "";
    prompt = "once upon a time,";     
    mt = ModelType::MODEL_LLAMA;  
    bt = BackendType::CPU; 
    enableQuantization = false;
    exportQuantizedModel = false;  
}


int main(int argc, char* argv[]){

    init();
    parse(argc, argv);

    CInfer infer;
    infer.build(modelPath, tknzrPath, mt, bt, enableQuantization, exportQuantizedModel, exportFilePath);
    infer.generate(prompt);
}
