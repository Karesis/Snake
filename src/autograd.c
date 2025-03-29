#include "../include/autograd.h"
#include "../include/utils.h"
#include <string.h>

// 全局自动求导上下文
AutogradContext autograd_ctx = {
    .is_training = 1,
    .retain_graph = 0
};

// 设置梯度启用状态
void set_grad_enabled(int mode) {
    autograd_ctx.is_training = mode;
}

// 获取梯度启用状态
int is_grad_enabled(void) {
    return autograd_ctx.is_training;
}

// 设置是否需要梯度
void set_requires_grad(Tensor* tensor, int requires_grad) {
    if (!tensor) return;
    tensor->requires_grad = requires_grad;
}

// 保留梯度
void retain_grad(Tensor* tensor) {
    if (!tensor) return;
    tensor->requires_grad = 1;
}

// 清零梯度
void zero_grad(Tensor* tensor) {
    if (!tensor || !tensor->grad) return;
    
    #pragma omp parallel for
    for (size_t i = 0; i < tensor->storage->size; i++) {
        tensor->grad[i] = 0.0f;
    }
}

// 累积梯度
void accumulate_grad(Tensor* tensor, const Tensor* grad) {
    if (!tensor || !grad) return;
    
    if (!tensor->grad) {
        tensor->grad = safe_malloc(tensor->storage->size * sizeof(float));
        memset(tensor->grad, 0, tensor->storage->size * sizeof(float));
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < tensor->storage->size; i++) {
        tensor->grad[i] += grad->storage->data[i];
    }
}

// 反向传播
void backward(Tensor* tensor) {
    if (!tensor || !tensor->requires_grad) return;
    
    // 创建初始梯度
    if (!tensor->grad) {
        tensor->grad = safe_malloc(tensor->storage->size * sizeof(float));
        memset(tensor->grad, 0, tensor->storage->size * sizeof(float));
        tensor->grad[0] = 1.0f;  // 设置初始梯度为1
    }
    
    // 遍历所有父节点
    for (int i = 0; i < tensor->num_parents; i++) {
        Tensor* parent = tensor->parents[i];
        if (!parent || !parent->requires_grad) continue;
        
        // 根据操作类型计算梯度
        if (strcmp(tensor->op_name, "add") == 0) {
            accumulate_grad(parent, tensor->grad);
        }
        else if (strcmp(tensor->op_name, "mul") == 0) {
            // 对于乘法，需要计算另一个操作数的梯度
            Tensor* other = (i == 0) ? tensor->parents[1] : tensor->parents[0];
            if (other) {
                Tensor* grad = tensor_create(other->num_dims, other->dims, NULL);
                #pragma omp parallel for
                for (size_t j = 0; j < grad->storage->size; j++) {
                    grad->storage->data[j] = tensor->grad[j] * other->storage->data[j];
                }
                accumulate_grad(parent, grad);
                tensor_free(grad);
            }
        }
        else if (strcmp(tensor->op_name, "matmul") == 0) {
            // 矩阵乘法的梯度计算
            Tensor* other = (i == 0) ? tensor->parents[1] : tensor->parents[0];
            if (other) {
                Tensor* grad = tensor_create(other->num_dims, other->dims, NULL);
                if (i == 0) {
                    // 计算第一个矩阵的梯度
                    #pragma omp parallel for collapse(2)
                    for (int m = 0; m < parent->dims[0]; m++) {
                        for (int k = 0; k < parent->dims[1]; k++) {
                            float sum = 0.0f;
                            for (int n = 0; n < tensor->dims[1]; n++) {
                                sum += tensor->grad[m * tensor->dims[1] + n] * 
                                      other->storage->data[k * other->dims[1] + n];
                            }
                            grad->storage->data[m * parent->dims[1] + k] = sum;
                        }
                    }
                } else {
                    // 计算第二个矩阵的梯度
                    #pragma omp parallel for collapse(2)
                    for (int k = 0; k < parent->dims[0]; k++) {
                        for (int n = 0; n < parent->dims[1]; n++) {
                            float sum = 0.0f;
                            for (int m = 0; m < tensor->dims[0]; m++) {
                                sum += tensor->grad[m * tensor->dims[1] + n] * 
                                      other->storage->data[m * other->dims[1] + k];
                            }
                            grad->storage->data[k * parent->dims[1] + n] = sum;
                        }
                    }
                }
                accumulate_grad(parent, grad);
                tensor_free(grad);
            }
        }
    }
    
    // 如果不是保留图，则释放梯度
    if (!autograd_ctx.retain_graph) {
        free(tensor->grad);
        tensor->grad = NULL;
    }
}

// 无梯度上下文管理
static int no_grad_count = 0;

void no_grad_push(void) {
    no_grad_count++;
    if (no_grad_count == 1) {
        autograd_ctx.is_training = 0;
    }
}

void no_grad_pop(void) {
    no_grad_count--;
    if (no_grad_count == 0) {
        autograd_ctx.is_training = 1;
    }
}
