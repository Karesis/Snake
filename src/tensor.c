#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdarg.h>
#include <omp.h>
#include "../include/tensor.h"
#include "../include/utils.h"
#include <math.h>

// 创建存储
static Storage* storage_create(size_t size, const float* data) {
    Storage* storage = safe_malloc(sizeof(Storage));
    storage->size = size;
    storage->ref_count = 1;
    
    if (data) {
        storage->data = safe_malloc(size * sizeof(float));
        memcpy(storage->data, data, size * sizeof(float));
        storage->is_allocated = 1;
    } else {
        storage->data = safe_calloc(size, sizeof(float));
        storage->is_allocated = 1;
    }
    
    return storage;
}

// 释放存储
static void storage_free(Storage* storage) {
    if (!storage) return;
    
    storage->ref_count--;
    if (storage->ref_count <= 0) {
        if (storage->is_allocated) {
            free(storage->data);
        }
        free(storage);
    }
}

// 创建张量
Tensor* tensor_create(int num_dims, const int* dims, const float* data) {
    if (num_dims <= 0 || !dims) {
        report_error("Invalid dimensions");
        return NULL;
    }
    
    Tensor* tensor = safe_malloc(sizeof(Tensor));
    tensor->num_dims = num_dims;
    tensor->dims = safe_malloc(num_dims * sizeof(int));
    tensor->strides = safe_malloc(num_dims * sizeof(size_t));
    
    // 复制维度
    memcpy(tensor->dims, dims, num_dims * sizeof(int));
    
    // 计算总元素数和步长
    size_t total_elements = 1;
    for (int i = num_dims - 1; i >= 0; i--) {
        tensor->strides[i] = total_elements;
        total_elements *= dims[i];
    }
    
    // 创建存储
    tensor->storage = storage_create(total_elements, data);
    
    // 初始化其他字段
    tensor->requires_grad = 0;
    tensor->grad = NULL;
    tensor->is_leaf = 1;
    tensor->parents = NULL;
    tensor->num_parents = 0;
    tensor->op_name = NULL;
    
    return tensor;
}

// 创建零张量
Tensor* tensor_zeros(int num_dims, const int* dims) {
    return tensor_create(num_dims, dims, NULL);
}

// 创建单位张量
Tensor* tensor_ones(int num_dims, const int* dims) {
    Tensor* tensor = tensor_create(num_dims, dims, NULL);
    for (size_t i = 0; i < tensor->storage->size; i++) {
        tensor->storage->data[i] = 1.0f;
    }
    return tensor;
}

// 释放张量
void tensor_free(Tensor* tensor) {
    if (!tensor) return;
    
    // 释放梯度
    if (tensor->grad) {
        free(tensor->grad);
        tensor->grad = NULL;
    }
    
    // 释放父节点数组
    if (tensor->parents) {
        free(tensor->parents);
        tensor->parents = NULL;
    }
    
    // 释放维度数组
    if (tensor->dims) {
        free(tensor->dims);
        tensor->dims = NULL;
    }
    
    // 释放步长数组
    if (tensor->strides) {
        free(tensor->strides);
        tensor->strides = NULL;
    }
    
    // 释放存储
    if (tensor->storage) {
        storage_free(tensor->storage);
        tensor->storage = NULL;
    }
    
    free(tensor);
}

// 获取总元素数
size_t tensor_numel(const Tensor* tensor) {
    return tensor->storage->size;
}

// 填充张量
void tensor_fill_(Tensor* tensor, float value) {
    #pragma omp parallel for
    for (size_t i = 0; i < tensor->storage->size; i++) {
        tensor->storage->data[i] = value;
    }
}

// 清零张量
void tensor_zero_(Tensor* tensor) {
    tensor_fill_(tensor, 0.0f);
}

// 克隆张量
Tensor* tensor_clone(Tensor* tensor) {
    if (!tensor) return NULL;
    
    // 创建新的张量
    Tensor* clone = tensor_create(tensor->num_dims, tensor->dims, tensor->storage->data);
    
    // 复制其他属性
    clone->requires_grad = tensor->requires_grad;
    clone->is_leaf = tensor->is_leaf;
    clone->op_name = tensor->op_name ? strdup(tensor->op_name) : NULL;
    
    // 复制梯度
    if (tensor->grad) {
        clone->grad = safe_malloc(tensor->storage->size * sizeof(float));
        memcpy(clone->grad, tensor->grad, tensor->storage->size * sizeof(float));
    }
    
    return clone;
}

// 重塑张量
Tensor* tensor_reshape(Tensor* tensor, int num_dims, const int* new_dims) {
    size_t total_elements = 1;
    for (int i = 0; i < num_dims; i++) {
        total_elements *= new_dims[i];
    }
    
    if (total_elements != tensor->storage->size) {
        report_error("Invalid reshape dimensions");
        return NULL;
    }
    
    Tensor* reshaped = safe_malloc(sizeof(Tensor));
    reshaped->num_dims = num_dims;
    reshaped->dims = safe_malloc(num_dims * sizeof(int));
    reshaped->strides = safe_malloc(num_dims * sizeof(size_t));
    
    // 复制维度
    memcpy(reshaped->dims, new_dims, num_dims * sizeof(int));
    
    // 计算步长
    size_t stride = 1;
    for (int i = num_dims - 1; i >= 0; i--) {
        reshaped->strides[i] = stride;
        stride *= new_dims[i];
    }
    
    // 共享存储
    reshaped->storage = tensor->storage;
    reshaped->storage->ref_count++;
    
    // 初始化其他字段
    reshaped->requires_grad = tensor->requires_grad;
    reshaped->grad = NULL;
    reshaped->is_leaf = tensor->is_leaf;
    reshaped->parents = NULL;
    reshaped->num_parents = 0;
    reshaped->op_name = NULL;
    
    return reshaped;
}

// 视图操作
Tensor* tensor_view(Tensor* tensor, int num_dims, const int* new_dims) {
    return tensor_reshape(tensor, num_dims, new_dims);
}

// 递归打印辅助函数
static void print_tensor_recursive(const Tensor* tensor, const size_t* strides, int depth, size_t offset) {
    if (depth == tensor->num_dims) {
        printf("%.4f", tensor->storage->data[offset]);
        return;
    }
    
    printf("[");
    
    // 对于非最后两个维度，添加换行和缩进
    if (depth < tensor->num_dims - 2) {
        printf("\n");
        for (int j = 0; j <= depth; j++) printf("  ");
    }
    
    for (int i = 0; i < tensor->dims[depth]; i++) {
        print_tensor_recursive(tensor, strides, depth + 1, offset + i * strides[depth]);
        
        if (i < tensor->dims[depth] - 1) {
            if (depth == tensor->num_dims - 2) {
                printf("\n");  // 倒数第二维度只用换行分隔
                for (int j = 0; j < depth; j++) printf("  ");
            } else if (depth == tensor->num_dims - 1) {
                printf(", ");  // 最后一维度用逗号和空格分隔
            } else {
                printf(",\n");  // 其他维度用逗号和换行分隔
                for (int j = 0; j <= depth; j++) printf("  ");
            }
        }
    }
    
    // 对于非最后两个维度，在结束时添加换行和缩进
    if (depth < tensor->num_dims - 2) {
        printf("\n");
        for (int j = 0; j < depth; j++) printf("  ");
    }
    printf("]");
}

// 打印张量
void tensor_print(Tensor* tensor) {
    if (!tensor) {
        printf("NULL tensor\n");
        return;
    }
    
    // 计算总元素数和每个维度的步长
    size_t* strides = safe_malloc(tensor->num_dims * sizeof(size_t));
    size_t stride = 1;
    for (int i = tensor->num_dims - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= tensor->dims[i];
    }
    
    print_tensor_recursive(tensor, strides, 0, 0);
    
    // 打印维度信息
    printf("\nshape: (");
    for (int i = 0; i < tensor->num_dims; i++) {
        printf("%d", tensor->dims[i]);
        if (i < tensor->num_dims - 1) printf(", ");
    }
    printf(")\n");
    
    free(strides);
}

// 设置父节点
static void set_parents(Tensor* result, const Tensor* a, const Tensor* b, const char* op_name) {
    result->requires_grad = a->requires_grad || b->requires_grad;
    result->is_leaf = 0;
    result->op_name = op_name;
    
    if (result->requires_grad) {
        result->parents = safe_malloc(2 * sizeof(Tensor*));
        result->parents[0] = (Tensor*)a;
        result->parents[1] = (Tensor*)b;
        result->num_parents = 2;
    }
}

// 张量加法
Tensor* tensor_add(const Tensor* a, const Tensor* b) {
    if (a->num_dims != b->num_dims) {
        report_error("Dimension mismatch in add");
        return NULL;
    }
    
    for (int i = 0; i < a->num_dims; i++) {
        if (a->dims[i] != b->dims[i]) {
            report_error("Shape mismatch in add");
            return NULL;
        }
    }
    
    Tensor* result = tensor_create(a->num_dims, a->dims, NULL);
    
    #pragma omp parallel for
    for (size_t i = 0; i < result->storage->size; i++) {
        result->storage->data[i] = a->storage->data[i] + b->storage->data[i];
    }
    
    set_parents(result, a, b, "add");
    return result;
}

// 张量减法
Tensor* tensor_sub(const Tensor* a, const Tensor* b) {
    // 检查维度是否可以广播
    if (a->num_dims < b->num_dims) {
        report_error("First tensor has fewer dimensions than second tensor");
        return NULL;
    }
    
    // 创建结果张量，使用第一个张量的维度
    Tensor* result = tensor_create(a->num_dims, a->dims, NULL);
    
    // 计算步长
    size_t a_stride = 1;
    size_t b_stride = 1;
    for (int i = a->num_dims - 1; i >= 0; i--) {
        if (i >= a->num_dims - b->num_dims) {
            int b_dim = b->dims[i - (a->num_dims - b->num_dims)];
            if (b_dim != 1 && b_dim != a->dims[i]) {
                report_error("Incompatible dimensions for broadcasting");
                tensor_free(result);
                return NULL;
            }
        }
    }
    
    // 执行广播减法
    #pragma omp parallel for
    for (size_t i = 0; i < result->storage->size; i++) {
        size_t b_index = i;
        if (b->storage->size == 1) {
            b_index = 0;  // 标量广播
        } else if (b->storage->size < a->storage->size) {
            // 计算广播索引
            b_index = 0;
            size_t temp = i;
            for (int j = a->num_dims - 1; j >= 0; j--) {
                if (j >= a->num_dims - b->num_dims) {
                    int b_dim = b->dims[j - (a->num_dims - b->num_dims)];
                    if (b_dim == 1) {
                        temp /= a->dims[j];
                    } else {
                        b_index = b_index * b_dim + (temp % a->dims[j]);
                        temp /= a->dims[j];
                    }
                }
            }
        }
        
        result->storage->data[i] = a->storage->data[i] - b->storage->data[b_index];
    }
    
    set_parents(result, a, b, "sub");
    return result;
}

// 张量乘法
Tensor* tensor_mul(const Tensor* a, const Tensor* b) {
    if (a->num_dims != b->num_dims) {
        report_error("Dimension mismatch in mul");
        return NULL;
    }
    
    for (int i = 0; i < a->num_dims; i++) {
        if (a->dims[i] != b->dims[i]) {
            report_error("Shape mismatch in mul");
            return NULL;
        }
    }
    
    Tensor* result = tensor_create(a->num_dims, a->dims, NULL);
    
    #pragma omp parallel for
    for (size_t i = 0; i < result->storage->size; i++) {
        result->storage->data[i] = a->storage->data[i] * b->storage->data[i];
    }
    
    set_parents(result, a, b, "mul");
    return result;
}

// 张量除法
Tensor* tensor_div(const Tensor* a, const Tensor* b) {
    if (a->num_dims != b->num_dims) {
        report_error("Dimension mismatch in div");
        return NULL;
    }
    
    for (int i = 0; i < a->num_dims; i++) {
        if (a->dims[i] != b->dims[i]) {
            report_error("Shape mismatch in div");
            return NULL;
        }
    }
    
    Tensor* result = tensor_create(a->num_dims, a->dims, NULL);
    
    int has_zero = 0;
    #pragma omp parallel for
    for (size_t i = 0; i < result->storage->size; i++) {
        if (b->storage->data[i] == 0) {
            has_zero = 1;
        }
        result->storage->data[i] = a->storage->data[i] / b->storage->data[i];
    }
    
    if (has_zero) {
        report_error("Division by zero");
        tensor_free(result);
        return NULL;
    }
    
    set_parents(result, a, b, "div");
    return result;
}

// 矩阵乘法
Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
    if (a->num_dims != 2 || b->num_dims != 2) {
        report_error("Matmul requires 2D tensors");
        return NULL;
    }
    
    if (a->dims[1] != b->dims[0]) {
        report_error("Invalid dimensions for matmul");
        return NULL;
    }
    
    int m = a->dims[0];
    int k = a->dims[1];
    int n = b->dims[1];
    
    int dims[2] = {m, n};
    Tensor* result = tensor_create(2, dims, NULL);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k_idx = 0; k_idx < k; k_idx++) {
                sum += a->storage->data[i * k + k_idx] * b->storage->data[k_idx * n + j];
            }
            result->storage->data[i * n + j] = sum;
        }
    }
    
    set_parents(result, a, b, "matmul");
    return result;
}

// 就地操作版本
void tensor_add_(Tensor* a, const Tensor* b) {
    if (a->num_dims != b->num_dims) {
        report_error("Dimension mismatch in add_");
        return;
    }
    
    for (int i = 0; i < a->num_dims; i++) {
        if (a->dims[i] != b->dims[i]) {
            report_error("Shape mismatch in add_");
            return;
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < a->storage->size; i++) {
        a->storage->data[i] += b->storage->data[i];
    }
}

void tensor_sub_(Tensor* a, const Tensor* b) {
    if (a->num_dims != b->num_dims) {
        report_error("Dimension mismatch in sub_");
        return;
    }
    
    for (int i = 0; i < a->num_dims; i++) {
        if (a->dims[i] != b->dims[i]) {
            report_error("Shape mismatch in sub_");
            return;
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < a->storage->size; i++) {
        a->storage->data[i] -= b->storage->data[i];
    }
}

void tensor_mul_(Tensor* a, const Tensor* b) {
    if (a->num_dims != b->num_dims) {
        report_error("Dimension mismatch in mul_");
        return;
    }
    
    for (int i = 0; i < a->num_dims; i++) {
        if (a->dims[i] != b->dims[i]) {
            report_error("Shape mismatch in mul_");
            return;
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < a->storage->size; i++) {
        a->storage->data[i] *= b->storage->data[i];
    }
}

void tensor_div_(Tensor* a, const Tensor* b) {
    if (a->num_dims != b->num_dims) {
        report_error("Dimension mismatch in div_");
        return;
    }
    
    for (int i = 0; i < a->num_dims; i++) {
        if (a->dims[i] != b->dims[i]) {
            report_error("Shape mismatch in div_");
            return;
        }
    }
    
    int has_zero = 0;
    #pragma omp parallel for
    for (size_t i = 0; i < a->storage->size; i++) {
        if (b->storage->data[i] == 0) {
            has_zero = 1;
        }
        a->storage->data[i] /= b->storage->data[i];
    }
    
    if (has_zero) {
        report_error("Division by zero");
    }
}

// 设置是否需要梯度
void set_requires_grad(Tensor* tensor, int requires_grad) {
    if (!tensor) return;
    tensor->requires_grad = requires_grad;
    tensor->is_leaf = !requires_grad;
}

// 累加梯度
void accumulate_grad(Tensor* tensor, Tensor* grad) {
    if (!tensor || !tensor->requires_grad || !grad) return;
    
    // 确保梯度内存已分配
    if (!tensor->grad) {
        tensor->grad = safe_calloc(tensor->storage->size, sizeof(float));
    }
    
    // 检查维度是否匹配
    if (tensor->storage->size != grad->storage->size) {
        report_error("Gradient dimensions do not match");
        return;
    }
    
    // 累加梯度
    #pragma omp parallel for
    for (size_t i = 0; i < tensor->storage->size; i++) {
        tensor->grad[i] += grad->storage->data[i];
    }
}

// 清零梯度
void zero_grad(Tensor* tensor) {
    if (!tensor || !tensor->requires_grad) return;
    
    // 如果梯度不存在，分配内存
    if (!tensor->grad) {
        tensor->grad = safe_calloc(tensor->storage->size, sizeof(float));
    }
    
    // 清零梯度
    memset(tensor->grad, 0, tensor->storage->size * sizeof(float));
}
