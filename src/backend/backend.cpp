#include <cmath>
#include "backend.hpp"

CBackend::CBackend(){

};
CBackend::~CBackend(){
    
};

void CBackend::softmax(float* x, int n) {
    // 找到最大值（数值稳定性）
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // 计算 exp(x[i] - max_val) 并求和
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // 归一化
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

void CBackend::matmul(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += w[i * n + j] * x[j];
        }
        xout[i] = sum;
    }
}

void CBackend::rmsnorm(float* o, float* x, float* weight, int size) {
    const float eps = 1e-5f;
    float sum = 0.0f;

    for (int i = 0; i < size; i++) {
        sum += x[i] * x[i];
    }

    float mean = sum / size;
    float scale = 1.0f / sqrtf(mean + eps);

    for (int i = 0; i < size; i++) {
        o[i] = x[i] * scale * weight[i];
    }
}



void CBackend::axpy(float *y, float *x, float factor, int dim) {
    for (int i = 0; i < dim; ++i) {
        y[i] += x[i] * factor;
    }
}

void CBackend::dot(float *y, float *x1, float*x2, int dim) {
    float y_tmp = 0.0;
    for (int i = 0; i < dim; ++i) {
        y_tmp += x1[i] * x2[i];
    }
    *y += y_tmp;
}

void CBackend::ropeEncoding(float *q, float *k, int headSize, int position, int dim, int kvDim) {
    for (int i = 0; i < dim; i+=2) {
        int headDim = i % headSize;
        float freq = 1.0f / powf(10000.0f, headDim / (float)headSize);
        float val = position * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kvDim ? 2 : 1; 
        for (int v = 0; v < rotn; v++) {
            float *vec = v == 0 ? q : k;
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }
}
void CBackend::gemvQkSeq(float *q, float *key, float *attentionScores, int pos, int kvDim, int headSize) {
    for (int timestep = 0; timestep <= pos; timestep++) {
        float* k = key + timestep * kvDim;
        float score = 0.0f;
        dot(&score, q, k, headSize);
        score /= sqrtf(headSize);
        attentionScores[timestep] = score;
    }
}

void CBackend::weightedV(float *headOutput, float *value, float *attentionScores, int pos, int kvDim, int headSize) {
    for (int t = 0; t <= pos; t++) {
        float *v = value + t * kvDim;
        float a = attentionScores[t];
        axpy(headOutput, v, a, headSize);
    }
}
void CBackend::swiGLLUFunc(float *headOutput, float *value, int hiddenDim) {
    for (int i = 0; i < hiddenDim; i++) {
        float val = headOutput[i];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= value[i];
        headOutput[i] = val;
    }
}
