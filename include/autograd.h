#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"

// 自动求导上下文
typedef struct {
    int is_training;
    int retain_graph;
} AutogradContext;

// 全局自动求导上下文
extern AutogradContext autograd_ctx;

// 自动求导函数
void backward(Tensor* tensor);
void retain_grad(Tensor* tensor);
void set_requires_grad(Tensor* tensor, int requires_grad);

// 梯度计算函数
void zero_grad(Tensor* tensor);
void accumulate_grad(Tensor* tensor, const Tensor* grad);

// 上下文管理
void set_grad_enabled(int mode);
int is_grad_enabled(void);
void no_grad_push(void);
void no_grad_pop(void);

#endif // AUTOGRAD_H
