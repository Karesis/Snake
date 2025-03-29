#ifndef NN_H
#define NN_H

#include "tensor.h"

// 前向传播函数指针类型
typedef struct Module Module;
typedef Tensor* (*ForwardFunc)(Module* self, Tensor* input);
typedef void (*BackwardFunc)(Module* self, Tensor* grad_output);
typedef void (*ZeroGradFunc)(Module* self);
typedef void (*FreeFunc)(Module* self);

// 基础模块结构
typedef struct Module {
    char name[32];
    int training;
    Tensor** parameters;
    int num_parameters;
    Module** children;
    int num_children;
    Tensor* input;  // 保存输入用于反向传播
    ForwardFunc forward;
    BackwardFunc backward;
    ZeroGradFunc zero_grad;
    FreeFunc free;
} Module;

// 层类型
typedef enum {
    LAYER_LINEAR,
    LAYER_RELU,
    LAYER_SIGMOID,
    LAYER_TANH,
    LAYER_SEQUENTIAL
} LayerType;

// 线性层
typedef struct {
    Module base;
    int in_features;
    int out_features;
    Tensor* weight;
    Tensor* bias;
} LinearLayer;

// ReLU层
typedef struct {
    Module base;
} ReLULayer;

// Sigmoid层
typedef struct {
    Module base;
} SigmoidLayer;

// Tanh层
typedef struct {
    Module base;
} TanhLayer;

// 顺序层
typedef struct {
    Module base;
    Module** layers;
    int num_layers;
} SequentialLayer;

// 创建函数
LinearLayer* nn_linear(int in_features, int out_features, int bias);
ReLULayer* nn_relu(void);
SigmoidLayer* nn_sigmoid(void);
TanhLayer* nn_tanh(void);
SequentialLayer* nn_sequential(Module** layers, int num_layers);

// 前向传播
void module_forward(Module* module, Tensor* input, Tensor* output);
void module_backward(Module* module, Tensor* grad_output, Tensor* grad_input);

// 梯度清零
void module_zero_grad(Module* module);

// 释放函数
void module_free(Module* module);

#endif // NN_H
