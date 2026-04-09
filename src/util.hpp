
#ifndef UTIL_HPP
#define UTIL_HPP
enum ModelType {
    MODEL_LLAMA       = 0,
    MODEL_DEEPSEEK    = 1,
};

enum BackendType {
    CPU         = 0,
    CPU_X86     = 1,
    CPU_ARM     = 2, 
    CUDA        = 20,
};

#endif // UTIL_HPP