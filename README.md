# Snake - 轻量级C语言深度学习框架

Snake是一个用C语言编写的轻量级深度学习框架，提供了张量运算、神经网络层和优化器等基本功能。该框架设计简单直观，适合用于学习和理解深度学习的基本概念。

## 功能特性

- 张量运算
  - 基本数学运算（加法、减法、乘法、除法）
  - 矩阵乘法
  - 张量重塑和视图操作
  - 自动求导支持

- 神经网络层
  - 线性层（Linear）
  - ReLU激活函数
  - Sigmoid激活函数
  - Tanh激活函数
  - Sequential容器

- 优化器
  - SGD（随机梯度下降）
  - Adam优化器

## 内存管理说明

由于使用C语言编写，Snake需要手动管理内存。以下是重要的内存管理规则：

### 1. 张量内存管理
```c
// 创建张量
Tensor* tensor = tensor_create(2, dims, data);

// 使用张量
// ...

// 释放张量
tensor_free(tensor);
```

注意事项：
- 每个`tensor_create`调用都需要对应的`tensor_free`
- 使用`tensor_clone`创建的张量也需要手动释放
- 张量的梯度内存会自动管理，不需要手动释放

### 2. 神经网络层内存管理
```c
// 创建层
LinearLayer* linear = nn_linear(in_features, out_features, bias);
ReLULayer* relu = nn_relu();

// 使用层
// ...

// 释放层
module_free((Module*)linear);
module_free((Module*)relu);
```

注意事项：
- 每个层创建函数都需要对应的`module_free`
- `module_free`会自动释放层内的所有参数和梯度

### 3. Sequential容器内存管理
```c
// 创建层数组
Module** layers = safe_malloc(2 * sizeof(Module*));
layers[0] = (Module*)linear;
layers[1] = (Module*)relu;

// 创建Sequential容器
SequentialLayer* seq = nn_sequential(layers, 2);

// 使用容器
// ...

// 释放内存
module_free((Module*)seq);  // 这会释放所有子层
free(layers);  // 释放层数组
```

注意事项：
- `module_free`会递归释放所有子层
- 需要手动释放层数组

### 4. 优化器内存管理
```c
// 创建优化器
SGD* sgd = optim_sgd(model, learning_rate, momentum, weight_decay);

// 使用优化器
// ...

// 释放优化器
sgd->base.free((Optimizer*)sgd);
```

注意事项：
- 优化器会管理自己的动量数组等内存
- 优化器不会释放模型的内存

## 使用示例

### 1. 基本张量操作
```c
// 创建张量
int dims[] = {2, 3};
float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
Tensor* t1 = tensor_create(2, dims, data);

// 创建另一个张量
float data2[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
Tensor* t2 = tensor_create(2, dims, data2);

// 执行运算
Tensor* sum = tensor_add(t1, t2);
Tensor* prod = tensor_matmul(t1, t2);

// 打印结果
tensor_print(sum);
tensor_print(prod);

// 释放内存
tensor_free(t1);
tensor_free(t2);
tensor_free(sum);
tensor_free(prod);
```

### 2. 创建和使用神经网络
```c
// 创建层
LinearLayer* linear = nn_linear(3, 2, 1);
ReLULayer* relu = nn_relu();

// 创建Sequential容器
Module** layers = safe_malloc(2 * sizeof(Module*));
layers[0] = (Module*)linear;
layers[1] = (Module*)relu;
SequentialLayer* model = nn_sequential(layers, 2);

// 创建输入
int input_dims[] = {1, 3};
float input_data[] = {0.1, 0.2, 0.3};
Tensor* input = tensor_create(2, input_dims, input_data);

// 前向传播
Tensor* output = model->base.forward((Module*)model, input);

// 创建优化器
SGD* optimizer = optim_sgd((Module*)model, 0.01, 0.0, 0.0);

// 反向传播和优化
model->base.backward((Module*)model, output);
optimizer->base.step((Optimizer*)optimizer);

// 释放内存
tensor_free(input);
tensor_free(output);
module_free((Module*)model);
optimizer->base.free((Optimizer*)optimizer);
free(layers);
```

## 编译和运行

```bash
# 编译
gcc -g -I. test_program.c src/tensor.c src/nn.c src/optim.c src/utils.c -lm -fopenmp -o test_program

# 运行测试
./test_program
```

## 注意事项

1. 内存泄漏检查
   - 使用Valgrind等工具检查内存泄漏
   - 确保每个分配的内存都有对应的释放

2. 线程安全
   - 框架使用OpenMP进行并行计算
   - 多线程环境下需要谨慎管理共享资源

3. 错误处理
   - 所有函数都会进行基本的错误检查
   - 建议在使用前检查返回值是否为NULL

4. 性能优化
   - 使用OpenMP进行并行计算
   - 避免频繁的内存分配和释放

## 许可证

MIT License 