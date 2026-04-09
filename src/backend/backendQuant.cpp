#include <cmath>
#include "backendQuant.hpp"
#include <iostream>
#include <stdint.h>
#include <cstring>
#include <vector>

void CBackendQuant::matmulQuant(float* o, float* x, float* w_quant, float* scale, int n, int d) {
    // 将 float* 指针转换为 int8* 来访问量化数据
    int8_t* w_quant_int8 = (int8_t*)w_quant;

    // 执行量化矩阵-向量乘法：o = w_quant * x
    // w_quant 是 d×n 的矩阵（按行存储），存储为 int8
    // x 是输入向量，大小为 n
    // o 是输出向量，大小为 d
    // scale 是全局缩放因子指针，用于反量化

    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            // 从量化矩阵中读取 int8 值
            int8_t w_quant_val = w_quant_int8[i * n + j];
            // 反量化：int8 值乘以全局缩放因子
            float w_float = (float)w_quant_val * (*scale);
            // 累加乘积
            sum += w_float * x[j];
        }
        o[i] = sum;
    }
}