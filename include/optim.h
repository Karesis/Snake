#ifndef OPTIM_H
#define OPTIM_H

#include "nn.h"
#include "tensor.h"

// 优化器基类
typedef struct Optimizer {
    Module* model;
    float lr;
    
    // 虚函数
    void (*step)(struct Optimizer* self);
    void (*zero_grad)(struct Optimizer* self);
    void (*free)(struct Optimizer* self);
} Optimizer;

// SGD优化器
typedef struct {
    Optimizer base;
    float momentum;
    float weight_decay;
    float** velocity;  // 动量
} SGD;

// Adam优化器
typedef struct {
    Optimizer base;
    float beta1;
    float beta2;
    float epsilon;
    float** m;  // 一阶矩估计
    float** v;  // 二阶矩估计
    int step_count;
} Adam;

// 创建函数
SGD* optim_sgd(Module* model, float lr, float momentum, float weight_decay);
Adam* optim_adam(Module* model, float lr, float beta1, float beta2, float epsilon);

#endif // OPTIM_H
