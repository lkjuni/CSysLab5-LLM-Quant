#ifndef BACKENDQUANT_HPP
#define BACKENDQUANT_HPP
#include "backend.hpp"
class CBackendQuant : public CBackend {
    public:
        void matmulQuant(float* o, float* x, float* w_quant, float* scale, int n, int d);
    };

#endif

