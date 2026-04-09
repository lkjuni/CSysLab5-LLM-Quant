#ifndef BACKEND_HPP
#define BACKEND_HPP
class CBackend {
    public:
    
       virtual void softmax(float* x, int size);
       virtual void matmul(float* xout, float* x, float* w, int n, int d);
       virtual void rmsnorm(float* o, float* x, float* weight, int size);
       virtual void axpy(float *y, float *x, float factor, int dim);
       virtual void dot(float *y, float *x1, float*x2, int dim);
       virtual void ropeEncoding(float *q, float *k, int headSize, int position, int dim, int kvDim);
       virtual void gemvQkSeq(float *q, float *key, float *att, int pos, int kvDim, int headSize);
       virtual void weightedV(float *xb, float *value, float *att, int pos, int kvDim, int headSize);
       virtual void swiGLLUFunc(float *hb, float *hb2, int hiddenDim);
        CBackend();
        ~CBackend() ;
        
    };

#endif

