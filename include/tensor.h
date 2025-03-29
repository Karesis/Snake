#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stddef.h>
#include <stdarg.h>

// 存储结构，用于管理实际的数据
typedef struct Storage {
    float* data;
    size_t size;
    int ref_count;
    int is_allocated;  // 标记数据是否是动态分配的
} Storage;

// 张量结构
typedef struct Tensor {
    Storage* storage;      // 数据存储
    int num_dims;         // 维度数
    int* dims;           // 各维度大小
    size_t* strides;     // 步长
    int requires_grad;    // 是否需要梯度
    float* grad;         // 梯度
    int is_leaf;         // 是否是叶子节点
    
    // 用于自动求导的字段
    struct Tensor** parents;     // 父节点
    int num_parents;      // 父节点数量
    const char* op_name;  // 操作名称
} Tensor;

// 创建和销毁
Tensor* tensor_create(int num_dims, const int* dims, const float* data);
Tensor* tensor_zeros(int num_dims, const int* dims);
Tensor* tensor_ones(int num_dims, const int* dims);
Tensor* tensor_clone(Tensor* tensor);
void tensor_free(Tensor* tensor);

// 基本操作
void tensor_print(Tensor* tensor);
Tensor* tensor_reshape(Tensor* tensor, int num_dims, const int* new_dims);
Tensor* tensor_view(Tensor* tensor, int num_dims, const int* new_dims);

// 数学运算
Tensor* tensor_add(const Tensor* a, const Tensor* b);
Tensor* tensor_sub(const Tensor* a, const Tensor* b);
Tensor* tensor_mul(const Tensor* a, const Tensor* b);
Tensor* tensor_div(const Tensor* a, const Tensor* b);
Tensor* tensor_matmul(const Tensor* a, const Tensor* b);

// 就地操作版本
void tensor_add_(Tensor* a, const Tensor* b);
void tensor_sub_(Tensor* a, const Tensor* b);
void tensor_mul_(Tensor* a, const Tensor* b);
void tensor_div_(Tensor* a, const Tensor* b);

// 实用函数
size_t tensor_numel(const Tensor* tensor);
void tensor_fill_(Tensor* tensor, float value);
void tensor_zero_(Tensor* tensor);

// 自动求导相关函数
void set_requires_grad(Tensor* tensor, int requires_grad);
void accumulate_grad(Tensor* tensor, Tensor* grad);
void zero_grad(Tensor* tensor);

#endif // TENSOR_H
