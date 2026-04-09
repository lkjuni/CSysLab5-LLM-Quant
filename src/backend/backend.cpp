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


void CBackend::matmul(float* o, float* x, float* w, int n, int d) {
    // 向量-矩阵乘法：o = x^T @ w
    // o 是输出向量，大小为 d
    // x 是输入向量，大小为 n
    // w 是权重矩阵，大小为 n×d（按行存储）

    for (int i = 0; i < d; i++) {
        o[i] = 0.0f;
        for (int j = 0; j < n; j++) {
            o[i] += x[j] * w[j * d + i];
        }
    }
}


void CBackend::rmsnorm(float* y, float* x, float* w, int n) {
    // RMS 归一化：y = (x / RMS(x)) * w
    // 其中 RMS(x) = sqrt(mean(x^2)) = sqrt(sum(x^2) / n)

    // 计算 sum(x^2)
    float sum_squared = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_squared += x[i] * x[i];
    }

    // 计算 RMS
    float rms = sqrtf(sum_squared / n + 1e-5f);  // 加上小的 epsilon 避免除以零

    // 应用归一化和权重缩放
    for (int i = 0; i < n; i++) {
        y[i] = (x[i] / rms) * w[i];
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
