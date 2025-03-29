#include "include/tensor.h"
#include "include/nn.h"
#include "include/optim.h"
#include "include/utils.h"
#include <stdio.h>

// 测试张量基本操作
void test_tensor_operations() {
    printf("\n=== 测试张量基本操作 ===\n");
    
    // 创建张量
    int dims[] = {2, 3};
    float data1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float data2[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    
    Tensor* t1 = tensor_create(2, dims, data1);
    Tensor* t2 = tensor_create(2, dims, data2);  // 使用相同的形状
    
    printf("张量1:\n");
    tensor_print(t1);
    printf("张量2:\n");
    tensor_print(t2);
    
    // 测试加法
    Tensor* sum = tensor_add(t1, t2);
    printf("加法结果:\n");
    tensor_print(sum);
    
    // 测试矩阵乘法
    int dims3[] = {3, 2};
    float data3[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    Tensor* t3 = tensor_create(2, dims3, data3);
    printf("张量3:\n");
    tensor_print(t3);
    
    Tensor* prod = tensor_matmul(t1, t3);
    printf("矩阵乘法结果 (2x3 @ 3x2):\n");
    tensor_print(prod);
    
    // 测试梯度
    set_requires_grad(t1, 1);
    Tensor* grad = tensor_create(2, dims, NULL);  // 创建全1梯度
    for (int i = 0; i < 6; i++) {
        grad->storage->data[i] = 1.0f;
    }
    accumulate_grad(t1, grad);
    printf("带梯度的张量:\n");
    tensor_print(t1);
    
    // 释放内存
    tensor_free(t1);
    tensor_free(t2);
    tensor_free(t3);
    tensor_free(sum);
    tensor_free(prod);
    tensor_free(grad);
}

// 测试神经网络层
void test_neural_network() {
    printf("\n=== 测试神经网络层 ===\n");
    
    // 创建输入张量 (batch_size=2, input_features=3)
    int input_dims[] = {2, 3};
    float input_data[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    Tensor* input = tensor_create(2, input_dims, input_data);
    printf("输入张量:\n");
    tensor_print(input);
    
    // 创建线性层 (in_features=3, out_features=2)
    LinearLayer* linear = nn_linear(3, 2, 1);
    printf("线性层前向传播:\n");
    Tensor* linear_output = ((Module*)linear)->forward((Module*)linear, input);
    tensor_print(linear_output);
    
    // 创建ReLU层
    ReLULayer* relu = nn_relu();
    printf("ReLU层前向传播:\n");
    Tensor* relu_output = ((Module*)relu)->forward((Module*)relu, linear_output);
    tensor_print(relu_output);
    
    // 创建Sequential容器
    Module** layers = safe_malloc(2 * sizeof(Module*));
    layers[0] = (Module*)linear;
    layers[1] = (Module*)relu;
    SequentialLayer* seq = nn_sequential(layers, 2);
    printf("Sequential容器前向传播:\n");
    Tensor* seq_output = ((Module*)seq)->forward((Module*)seq, input);
    tensor_print(seq_output);
    
    // 创建梯度张量
    int grad_dims[] = {2, 2};  // 与输出形状相同
    float grad_data[] = {1.0, 1.0, 1.0, 1.0};  // 简单的梯度
    Tensor* grad = tensor_create(2, grad_dims, grad_data);
    
    // 反向传播
    ((Module*)seq)->backward((Module*)seq, grad);
    printf("反向传播完成\n");
    
    // 释放内存
    tensor_free(input);
    tensor_free(linear_output);
    tensor_free(relu_output);
    tensor_free(seq_output);
    tensor_free(grad);
    module_free((Module*)seq);  // 这会释放所有子模块
    free(layers);  // 释放层数组
    
    printf("神经网络测试完成\n");
}

// 测试优化器
void test_optimizer() {
    printf("\n=== 测试优化器 ===\n");
    
    // 创建一个简单的线性层
    LinearLayer* model = nn_linear(2, 1, 1);
    
    // 设置参数的requires_grad
    set_requires_grad(model->weight, 1);
    if (model->bias) {
        set_requires_grad(model->bias, 1);
    }
    
    // 创建SGD优化器
    SGD* sgd = optim_sgd((Module*)model, 0.01, 0.0, 0.0);  // 不使用动量和权重衰减
    printf("SGD优化器创建成功\n");
    
    // 创建输入数据
    int input_dims[] = {1, 2};
    float input_data[] = {1.0, 2.0};
    Tensor* input = tensor_create(2, input_dims, input_data);
    printf("输入数据:\n");
    tensor_print(input);
    
    // 前向传播
    Tensor* output = ((Module*)model)->forward((Module*)model, input);
    printf("模型输出:\n");
    tensor_print(output);
    
    // 创建梯度张量
    int grad_dims[] = {1, 1};  // 与输出形状相同
    float grad_data[] = {1.0};  // 简单的梯度
    Tensor* grad = tensor_create(2, grad_dims, grad_data);
    
    // 反向传播
    ((Module*)model)->backward((Module*)model, grad);
    
    // 优化器步进
    sgd->base.step((Optimizer*)sgd);
    printf("优化器步进完成\n");
    
    // 再次前向传播
    Tensor* new_output = ((Module*)model)->forward((Module*)model, input);
    printf("更新后的输出:\n");
    tensor_print(new_output);
    
    // 释放内存
    tensor_free(input);
    tensor_free(output);
    tensor_free(grad);
    tensor_free(new_output);
    module_free((Module*)model);
    sgd->base.free((Optimizer*)sgd);
    
    printf("优化器测试完成\n");
}

int main() {
    printf("开始全面测试...\n");
    
    // 运行所有测试
    test_tensor_operations();
    test_neural_network();
    test_optimizer();
    
    printf("\n所有测试完成!\n");
    return 0;
} 