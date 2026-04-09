#include <cmath>
#include "backendQuant.hpp"
#include <iostream>
#include <stdint.h>
#include <cstring>
#include <vector>

void CBackendQuant::matmulQuant(float* o, float* x, float* w_quant, float* scale, int n, int d) { 
/**
* TODO： 执行量化矩阵-向量乘法（MatMul），其中 w_quant 以 int8 格式存储，并使用 scale 进行反量化
* @param[out] o  计算结果输出向量，大小为 d。计算公式： o(d,1) =  w_quant (d,n) × x(n,1)
* @param[in]  x  输入向量，大小为 n
* @param[in]  w_quant 量化权重矩阵，大小为 d * n，传入float* 指针， 存储为 int8 类型
* @param[in]  scale 反量化缩放因子
* @param[in]  n  输入向量 x 的维度（ w_quant 的列数）
* @param[in]  d  输出向量 o 的维度（ w_quant 的行数）
**/
}