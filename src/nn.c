#include "../include/nn.h"
#include "../include/utils.h"
#include <math.h>
#include <string.h>

// 线性层前向传播
static Tensor* linear_forward(Module* self, Tensor* input) {
    LinearLayer* linear = (LinearLayer*)self;
    
    // 保存输入用于反向传播
    linear->base.input = tensor_clone(input);
    
    // 转置权重矩阵
    int weight_t_dims[] = {linear->in_features, linear->out_features};
    Tensor* weight_t = tensor_create(2, weight_t_dims, NULL);
    
    // 手动转置
    for (int i = 0; i < linear->out_features; i++) {
        for (int j = 0; j < linear->in_features; j++) {
            weight_t->storage->data[j * linear->out_features + i] = 
                linear->weight->storage->data[i * linear->in_features + j];
        }
    }
    
    // 矩阵乘法
    Tensor* output = tensor_matmul(input, weight_t);
    tensor_free(weight_t);
    
    // 添加偏置
    if (linear->bias) {
        // 广播偏置到每一行
        for (int i = 0; i < output->dims[0]; i++) {
            for (int j = 0; j < output->dims[1]; j++) {
                output->storage->data[i * output->dims[1] + j] += linear->bias->storage->data[j];
            }
        }
    }
    
    return output;
}

// 线性层反向传播
static void linear_backward(Module* self, Tensor* grad_output) {
    LinearLayer* linear = (LinearLayer*)self;
    
    // 检查输入是否存在
    if (!linear->base.input) {
        report_error("No input tensor saved for backward pass");
        return;
    }
    
    // 计算权重的梯度
    if (linear->weight->requires_grad) {
        // 创建权重梯度张量
        int weight_grad_dims[] = {linear->out_features, linear->in_features};
        Tensor* weight_grad = tensor_create(2, weight_grad_dims, NULL);
        
        // 手动计算权重梯度：grad_output.T @ input
        for (int i = 0; i < linear->out_features; i++) {
            for (int j = 0; j < linear->in_features; j++) {
                float sum = 0.0f;
                for (int k = 0; k < grad_output->dims[0]; k++) {
                    sum += grad_output->storage->data[k * grad_output->dims[1] + i] * 
                           linear->base.input->storage->data[k * linear->in_features + j];
                }
                weight_grad->storage->data[i * linear->in_features + j] = sum;
            }
        }
        
        // 累加梯度
        accumulate_grad(linear->weight, weight_grad);
        
        // 释放内存
        tensor_free(weight_grad);
    }
    
    // 计算偏置的梯度
    if (linear->bias && linear->bias->requires_grad) {
        // 创建偏置梯度张量
        int bias_dims[] = {linear->out_features};
        Tensor* bias_grad = tensor_create(1, bias_dims, NULL);
        
        // 计算偏置梯度：对每一行的梯度求和
        for (int i = 0; i < linear->out_features; i++) {
            float sum = 0.0f;
            for (int j = 0; j < grad_output->dims[0]; j++) {
                sum += grad_output->storage->data[j * grad_output->dims[1] + i];
            }
            bias_grad->storage->data[i] = sum;
        }
        
        // 累加梯度
        accumulate_grad(linear->bias, bias_grad);
        
        // 释放内存
        tensor_free(bias_grad);
    }
}

// 线性层清零梯度
static void linear_zero_grad(Module* self) {
    LinearLayer* linear = (LinearLayer*)self;
    if (linear->weight) {
        zero_grad(linear->weight);
    }
    if (linear->bias) {
        zero_grad(linear->bias);
    }
}

// 线性层释放
static void linear_free(Module* self) {
    LinearLayer* linear = (LinearLayer*)self;
    if (linear->weight) {
        tensor_free(linear->weight);
    }
    if (linear->bias) {
        tensor_free(linear->bias);
    }
    if (linear->base.input) {
        tensor_free(linear->base.input);
    }
    free(linear);
}

// 创建线性层
LinearLayer* nn_linear(int in_features, int out_features, int bias) {
    LinearLayer* linear = safe_malloc(sizeof(LinearLayer));
    
    // 初始化基类
    strcpy(linear->base.name, "Linear");
    linear->base.training = 1;
    linear->base.children = NULL;
    linear->base.num_children = 0;
    linear->base.input = NULL;  // 初始化input字段
    
    // 初始化参数数组
    linear->base.num_parameters = bias ? 2 : 1;
    linear->base.parameters = safe_malloc(linear->base.num_parameters * sizeof(Tensor*));
    
    // 初始化权重
    int weight_dims[2] = {out_features, in_features};  // 修改权重矩阵维度
    linear->weight = tensor_create(2, weight_dims, NULL);
    set_requires_grad(linear->weight, 1);
    
    // 初始化权重数据
    for (int i = 0; i < out_features * in_features; i++) {
        linear->weight->storage->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    // 将权重添加到参数列表
    linear->base.parameters[0] = linear->weight;
    
    // 初始化偏置
    if (bias) {
        int bias_dims[1] = {out_features};
        linear->bias = tensor_create(1, bias_dims, NULL);
        set_requires_grad(linear->bias, 1);
        
        // 初始化偏置数据
        for (int i = 0; i < out_features; i++) {
            linear->bias->storage->data[i] = 0.0f;
        }
        
        // 将偏置添加到参数列表
        linear->base.parameters[1] = linear->bias;
    } else {
        linear->bias = NULL;
    }
    
    // 设置虚函数
    linear->base.forward = linear_forward;
    linear->base.backward = linear_backward;
    linear->base.zero_grad = linear_zero_grad;
    linear->base.free = linear_free;
    
    linear->in_features = in_features;
    linear->out_features = out_features;
    
    return linear;
}

// ReLU前向传播
static Tensor* relu_forward(Module* self, Tensor* input) {
    Tensor* output = tensor_clone(input);
    
    #pragma omp parallel for
    for (size_t i = 0; i < output->storage->size; i++) {
        if (output->storage->data[i] < 0) {
            output->storage->data[i] = 0;
        }
    }
    
    return output;
}

// ReLU反向传播
static void relu_backward(Module* self, Tensor* grad_output) {
    // ReLU的梯度计算在autograd中处理
}

// ReLU清零梯度
static void relu_zero_grad(Module* self) {
    // ReLU没有参数，不需要清零梯度
}

// ReLU释放
static void relu_free(Module* self) {
    free(self);
}

// 创建ReLU层
ReLULayer* nn_relu(void) {
    ReLULayer* relu = safe_malloc(sizeof(ReLULayer));
    
    // 初始化基类
    strcpy(relu->base.name, "ReLU");
    relu->base.training = 1;
    relu->base.children = NULL;
    relu->base.num_children = 0;
    relu->base.parameters = NULL;
    relu->base.num_parameters = 0;
    relu->base.input = NULL;  // 初始化input字段
    
    // 设置虚函数
    relu->base.forward = relu_forward;
    relu->base.backward = relu_backward;
    relu->base.zero_grad = relu_zero_grad;
    relu->base.free = relu_free;
    
    return relu;
}

// Sigmoid前向传播
static Tensor* sigmoid_forward(Module* self, Tensor* input) {
    Tensor* output = tensor_clone(input);
    
    #pragma omp parallel for
    for (size_t i = 0; i < output->storage->size; i++) {
        output->storage->data[i] = 1.0f / (1.0f + expf(-output->storage->data[i]));
    }
    
    return output;
}

// Sigmoid反向传播
static void sigmoid_backward(Module* self, Tensor* grad_output) {
    // Sigmoid的梯度计算在autograd中处理
}

// Sigmoid清零梯度
static void sigmoid_zero_grad(Module* self) {
    // Sigmoid没有参数，不需要清零梯度
}

// Sigmoid释放
static void sigmoid_free(Module* self) {
    free(self);
}

// 创建Sigmoid层
SigmoidLayer* nn_sigmoid(void) {
    SigmoidLayer* sigmoid = safe_malloc(sizeof(SigmoidLayer));
    
    // 初始化基类
    strcpy(sigmoid->base.name, "Sigmoid");
    sigmoid->base.training = 1;
    sigmoid->base.children = NULL;
    sigmoid->base.num_children = 0;
    sigmoid->base.parameters = NULL;
    sigmoid->base.num_parameters = 0;
    
    // 设置虚函数
    sigmoid->base.forward = sigmoid_forward;
    sigmoid->base.backward = sigmoid_backward;
    sigmoid->base.zero_grad = sigmoid_zero_grad;
    sigmoid->base.free = sigmoid_free;
    
    return sigmoid;
}

// Tanh前向传播
static Tensor* tanh_forward(Module* self, Tensor* input) {
    Tensor* output = tensor_clone(input);
    
    #pragma omp parallel for
    for (size_t i = 0; i < output->storage->size; i++) {
        output->storage->data[i] = tanhf(output->storage->data[i]);
    }
    
    return output;
}

// Tanh反向传播
static void tanh_backward(Module* self, Tensor* grad_output) {
    // Tanh的梯度计算在autograd中处理
}

// Tanh清零梯度
static void tanh_zero_grad(Module* self) {
    // Tanh没有参数，不需要清零梯度
}

// Tanh释放
static void tanh_free(Module* self) {
    free(self);
}

// 创建Tanh层
TanhLayer* nn_tanh(void) {
    TanhLayer* tanh = safe_malloc(sizeof(TanhLayer));
    
    // 初始化基类
    strcpy(tanh->base.name, "Tanh");
    tanh->base.training = 1;
    tanh->base.children = NULL;
    tanh->base.num_children = 0;
    tanh->base.parameters = NULL;
    tanh->base.num_parameters = 0;
    
    // 设置虚函数
    tanh->base.forward = tanh_forward;
    tanh->base.backward = tanh_backward;
    tanh->base.zero_grad = tanh_zero_grad;
    tanh->base.free = tanh_free;
    
    return tanh;
}

// Sequential前向传播
static Tensor* sequential_forward(Module* self, Tensor* input) {
    SequentialLayer* seq = (SequentialLayer*)self;
    Tensor* current = tensor_clone(input);  // 克隆输入以避免修改原始数据
    
    // 保存输入用于反向传播
    seq->base.input = tensor_clone(input);
    
    // 遍历所有层
    for (int i = 0; i < seq->num_layers; i++) {
        Module* layer = seq->layers[i];
        if (!layer) continue;
        
        // 执行当前层的前向传播
        Tensor* next = layer->forward(layer, current);
        
        // 释放上一层的输出（除了第一次的输入克隆）
        tensor_free(current);
        current = next;
    }
    
    return current;
}

// Sequential反向传播
static void sequential_backward(Module* self, Tensor* grad_output) {
    SequentialLayer* seq = (SequentialLayer*)self;
    Tensor* current_grad = grad_output;
    
    // 从后向前遍历所有层
    for (int i = seq->num_layers - 1; i >= 0; i--) {
        Module* layer = seq->layers[i];
        if (!layer) continue;
        
        // 对当前层执行反向传播
        if (layer->backward) {
            layer->backward(layer, current_grad);
        }
        
        // 如果不是第一层，需要计算前一层的梯度
        if (i > 0) {
            // 获取前一层的输入
            Module* prev_layer = seq->layers[i-1];
            if (prev_layer && prev_layer->input) {
                int* prev_dims = prev_layer->input->dims;
                int n_dims = prev_layer->input->num_dims;
                
                // 创建新的梯度张量
                Tensor* prev_grad = tensor_create(n_dims, prev_dims, NULL);
                memcpy(prev_grad->storage->data, current_grad->storage->data, 
                       tensor_numel(prev_grad) * sizeof(float));
                
                // 更新当前梯度
                if (current_grad != grad_output) {
                    tensor_free(current_grad);
                }
                current_grad = prev_grad;
            }
        }
    }
    
    // 清理最后的梯度
    if (current_grad != grad_output) {
        tensor_free(current_grad);
    }
}

// Sequential清零梯度
static void sequential_zero_grad(Module* self) {
    SequentialLayer* seq = (SequentialLayer*)self;
    
    for (int i = 0; i < seq->num_layers; i++) {
        seq->layers[i]->zero_grad(seq->layers[i]);
    }
}

// Sequential释放
static void sequential_free(Module* self) {
    SequentialLayer* seq = (SequentialLayer*)self;
    
    // 释放保存的输入
    if (seq->base.input) {
        tensor_free(seq->base.input);
    }
    
    // 释放子模块
    for (int i = 0; i < seq->num_layers; i++) {
        if (seq->layers[i]) {
            seq->layers[i]->free(seq->layers[i]);
        }
    }
    
    // 不需要释放layers数组，因为它是在外部分配的
    free(seq);
}

// 创建Sequential容器
SequentialLayer* nn_sequential(Module** layers, int num_layers) {
    SequentialLayer* seq = safe_malloc(sizeof(SequentialLayer));
    
    // 初始化基类
    strcpy(seq->base.name, "Sequential");
    seq->base.training = 1;
    seq->layers = layers;
    seq->num_layers = num_layers;
    seq->base.parameters = NULL;
    seq->base.num_parameters = 0;
    seq->base.input = NULL;  // 初始化input字段
    
    // 设置虚函数
    seq->base.forward = sequential_forward;
    seq->base.backward = sequential_backward;
    seq->base.zero_grad = sequential_zero_grad;
    seq->base.free = sequential_free;
    
    return seq;
}

// 通用模块函数
void module_train(Module* module, int mode) {
    if (!module) return;
    
    module->training = mode;
    
    // 递归设置子模块的训练模式
    for (int i = 0; i < module->num_children; i++) {
        module_train(module->children[i], mode);
    }
}

void module_eval(Module* module) {
    module_train(module, 0);
}

void module_zero_grad(Module* module) {
    if (!module) return;
    module->zero_grad(module);
}

void module_free(Module* module) {
    if (!module) return;
    module->free(module);
}
