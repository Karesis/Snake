#include "../include/optim.h"
#include "../include/utils.h"
#include "../include/nn.h"
#include <math.h>

// SGD优化器步进
static void sgd_step(Optimizer* self) {
    SGD* sgd = (SGD*)self;
    Module* model = sgd->base.model;
    
    if (!model || !model->parameters) {
        return;
    }
    
    // 遍历所有参数
    for (int i = 0; i < model->num_parameters; i++) {
        Tensor* param = model->parameters[i];
        if (!param || !param->requires_grad || !param->storage || !param->storage->data) continue;
        
        // 计算梯度
        float* grad = param->grad;
        if (!grad) continue;
        
        // 应用权重衰减
        if (sgd->weight_decay != 0.0f) {
            #pragma omp parallel for
            for (size_t j = 0; j < param->storage->size; j++) {
                grad[j] += sgd->weight_decay * param->storage->data[j];
            }
        }
        
        // 应用动量
        if (sgd->momentum != 0.0f) {
            if (!sgd->velocity[i]) {
                sgd->velocity[i] = safe_calloc(param->storage->size, sizeof(float));
            }
            
            #pragma omp parallel for
            for (size_t j = 0; j < param->storage->size; j++) {
                sgd->velocity[i][j] = sgd->momentum * sgd->velocity[i][j] - sgd->base.lr * grad[j];
                param->storage->data[j] += sgd->velocity[i][j];
            }
        } else {
            // 直接更新参数
            #pragma omp parallel for
            for (size_t j = 0; j < param->storage->size; j++) {
                param->storage->data[j] -= sgd->base.lr * grad[j];
            }
        }
        
        // 清零梯度
        zero_grad(param);
    }
}

// SGD优化器清零梯度
static void sgd_zero_grad(Optimizer* self) {
    if (!self || !self->model) return;
    module_zero_grad(self->model);
}

// SGD优化器释放
static void sgd_free(Optimizer* self) {
    if (!self) return;
    
    SGD* sgd = (SGD*)self;
    
    // 释放动量数组
    if (sgd->velocity) {
        for (int i = 0; i < sgd->base.model->num_parameters; i++) {
            if (sgd->velocity[i]) {
                free(sgd->velocity[i]);
            }
        }
        free(sgd->velocity);
    }
    
    free(sgd);
}

// 创建SGD优化器
SGD* optim_sgd(Module* model, float lr, float momentum, float weight_decay) {
    if (!model) return NULL;
    
    SGD* sgd = safe_malloc(sizeof(SGD));
    
    // 初始化基类
    sgd->base.model = model;
    sgd->base.lr = lr;
    sgd->base.step = sgd_step;
    sgd->base.zero_grad = sgd_zero_grad;
    sgd->base.free = sgd_free;
    
    // 初始化SGD特定参数
    sgd->momentum = momentum;
    sgd->weight_decay = weight_decay;
    sgd->velocity = safe_calloc(model->num_parameters, sizeof(float*));
    
    return sgd;
}

// Adam优化器步进
static void adam_step(Optimizer* self) {
    Adam* adam = (Adam*)self;
    Module* model = adam->base.model;
    
    if (!model || !model->parameters) {
        return;
    }
    
    adam->step_count++;
    float lr = adam->base.lr * sqrtf(1.0f - powf(adam->beta2, adam->step_count)) / 
               (1.0f - powf(adam->beta1, adam->step_count));
    
    // 遍历所有参数
    for (int i = 0; i < model->num_parameters; i++) {
        Tensor* param = model->parameters[i];
        if (!param || !param->requires_grad || !param->storage || !param->storage->data) continue;
        
        // 计算梯度
        float* grad = param->grad;
        if (!grad) continue;
        
        // 初始化动量数组
        if (!adam->m[i]) {
            adam->m[i] = safe_calloc(param->storage->size, sizeof(float));
            adam->v[i] = safe_calloc(param->storage->size, sizeof(float));
        }
        
        // 更新一阶矩估计
        #pragma omp parallel for
        for (size_t j = 0; j < param->storage->size; j++) {
            adam->m[i][j] = adam->beta1 * adam->m[i][j] + (1.0f - adam->beta1) * grad[j];
            adam->v[i][j] = adam->beta2 * adam->v[i][j] + (1.0f - adam->beta2) * grad[j] * grad[j];
            
            float m_hat = adam->m[i][j] / (1.0f - powf(adam->beta1, adam->step_count));
            float v_hat = adam->v[i][j] / (1.0f - powf(adam->beta2, adam->step_count));
            
            param->storage->data[j] -= lr * m_hat / (sqrtf(v_hat) + adam->epsilon);
        }
        
        // 清零梯度
        zero_grad(param);
    }
}

// Adam优化器清零梯度
static void adam_zero_grad(Optimizer* self) {
    if (!self || !self->model) return;
    module_zero_grad(self->model);
}

// Adam优化器释放
static void adam_free(Optimizer* self) {
    if (!self) return;
    
    Adam* adam = (Adam*)self;
    
    // 释放动量数组
    if (adam->m) {
        for (int i = 0; i < adam->base.model->num_parameters; i++) {
            if (adam->m[i]) {
                free(adam->m[i]);
            }
            if (adam->v[i]) {
                free(adam->v[i]);
            }
        }
        free(adam->m);
        free(adam->v);
    }
    
    free(adam);
}

// 创建Adam优化器
Adam* optim_adam(Module* model, float lr, float beta1, float beta2, float epsilon) {
    if (!model) return NULL;
    
    Adam* adam = safe_malloc(sizeof(Adam));
    
    // 初始化基类
    adam->base.model = model;
    adam->base.lr = lr;
    adam->base.step = adam_step;
    adam->base.zero_grad = adam_zero_grad;
    adam->base.free = adam_free;
    
    // 初始化Adam特定参数
    adam->beta1 = beta1;
    adam->beta2 = beta2;
    adam->epsilon = epsilon;
    adam->step_count = 0;
    adam->m = safe_calloc(model->num_parameters, sizeof(float*));
    adam->v = safe_calloc(model->num_parameters, sizeof(float*));
    
    return adam;
}
