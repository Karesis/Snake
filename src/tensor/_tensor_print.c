#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "tensor/_tensor_print.h"
#include "utils/_malloc.h" // 假设你有 safemalloc

// --- 内部辅助结构和函数 ---

// 模仿 PyTorch 的 PrintFormat 结构
typedef enum {
    FORMAT_DEFAULT,     // 对应 'g' 格式
    FORMAT_SCIENTIFIC,  // 对应 'e' 格式
    FORMAT_FIXED        // 对应 'f' 格式
} FormatType;

typedef struct {
    double scale;
    int width;
    int precision;
    FormatType type;
} PrintFormat;


// 将任何类型的 Tensor 数据转换为 double 数组，以便统一处理
// 返回 0 表示成功, -1 表示失败
static int _tensor_to_double_array(const Tensor t, double** out_data) {
    size_t n = shape_get_elements_count(t->shape);
    if (n == 0) {
        *out_data = NULL;
        return 0;
    }

    *out_data = (double*)safemalloc(n * sizeof(double));
    if (*out_data == NULL) return -1;

    // 根据你的 DataType enum 在这里添加转换逻辑
    // 这里只示例了 float 和 int
    switch (t->dtype) {
        case DTYPE_F32: {
            float* src = (float*)t->data;
            for (size_t i = 0; i < n; ++i) (*out_data)[i] = (double)src[i];
            break;
        }
        case DTYPE_I32: {
            int* src = (int*)t->data;
            for (size_t i = 0; i < n; ++i) (*out_data)[i] = (double)src[i];
            break;
        }
        case DTYPE_F64: { // 如果已经是 double，直接 memcpy
            memcpy(*out_data, t->data, n * sizeof(double));
            break;
        }
        // ... 在此添加其他数据类型的转换 ...
        default:
            fprintf(stderr, "Error: Unsupported dtype for printing.\n");
            free(*out_data);
            *out_data = NULL;
            return -1;
    }
    return 0;
}


// 模仿 PyTorch 的 __printFormat 函数，这是精髓！
static PrintFormat _calculate_print_format(const double* data, size_t n) {
    if (n == 0) {
        return (PrintFormat){1.0, 0, 4, FORMAT_DEFAULT};
    }

    // 检查是否所有数都可以被当作整数打印
    bool int_mode = true;
    for (size_t i = 0; i < n; ++i) {
        if (isfinite(data[i]) && data[i] != ceil(data[i])) {
            int_mode = false;
            break;
        }
    }

    // 计算最大和最小的指数 (10的幂)
    double exp_min = 1.0, exp_max = 1.0;
    size_t offset = 0;
    while(offset < n && !isfinite(data[offset])) {
        offset++;
    }

    if (offset == n) { // 全是 non-finite
        exp_min = 0.0; exp_max = 0.0;
    } else {
        exp_min = fabs(data[offset]);
        exp_max = fabs(data[offset]);
        for (size_t i = offset + 1; i < n; ++i) {
            double z = fabs(data[i]);
            if (isfinite(z)) {
                if (z < exp_min) exp_min = z;
                if (z > exp_max) exp_max = z;
            }
        }
    }

    if (exp_min != 0) exp_min = floor(log10(exp_min)) + 1;
    if (exp_max != 0) exp_max = floor(log10(exp_max)) + 1;


    if (int_mode) {
        if (exp_max > 9) {
            return (PrintFormat){1.0, 11, 4, FORMAT_SCIENTIFIC};
        } else {
            return (PrintFormat){1.0, (int)exp_max + 1, 0, FORMAT_DEFAULT};
        }
    } else {
        if (exp_max - exp_min > 4) {
            return (PrintFormat){1.0, 11, 4, FORMAT_SCIENTIFIC};
        } else {
            // 这里简化了 PyTorch 的逻辑，但效果接近
            if (exp_max > 5 || exp_max < 0) {
                 return (PrintFormat){1.0, (int)exp_max + 7, 4, FORMAT_FIXED};
            } else {
                 return (PrintFormat){1.0, (int)exp_max + 6, 4, FORMAT_FIXED};
            }
        }
    }
}

// 打印单个格式化后的值
static void _print_value(FILE* stream, double value, const PrintFormat* fmt) {
    double val = value / fmt->scale;
    switch (fmt->type) {
        case FORMAT_DEFAULT:
            fprintf(stream, "%*g", fmt->width, val);
            break;
        case FORMAT_SCIENTIFIC:
            fprintf(stream, "%*.*e", fmt->width, fmt->precision, val);
            break;
        case FORMAT_FIXED:
            fprintf(stream, "%*.*f", fmt->width, fmt->precision, val);
            break;
    }
}

// --- 核心打印函数 ---

// 打印2D矩阵
static void _print_matrix(FILE* stream, const double* data, const int* dims, int linesize, const PrintFormat* fmt, int indent) {
    long n_column_per_line = (linesize - indent) / (fmt->width + 1);
    if (n_column_per_line <= 0) n_column_per_line = 1;

    long first_col = 0;
    while (first_col < dims[1]) {
        long last_col = first_col + n_column_per_line - 1;
        if (last_col >= dims[1]) last_col = dims[1] - 1;

        if (n_column_per_line < dims[1]) {
             fprintf(stream, "%*sColumns %ld to %ld\n", indent, "", first_col + 1, last_col + 1);
        }

        for (int i = 0; i < dims[0]; ++i) { // rows
            fprintf(stream, "%*s", indent, "");
            for (long j = first_col; j <= last_col; ++j) { // cols
                _print_value(stream, data[i * dims[1] + j], fmt);
                if (j < last_col) {
                    fprintf(stream, " ");
                }
            }
            fprintf(stream, "\n");
        }
        first_col = last_col + 1;
        if (first_col < dims[1]) fprintf(stream, "\n");
    }
}

// 递归打印N-D张量
static void _print_tensor_recursive(FILE* stream, double** data_ptr, const int* dims, int ndim, int* counter, int linesize, const PrintFormat* fmt) {
    if (ndim == 2) {
        // 打印切片坐标
        fprintf(stream, "(");
        for(int i = 0; i < shape_get_ndim(counter) - 2; ++i) { // shape_get_ndim on what?? this is wrong
             fprintf(stream, "%d,", counter[i] + 1);
        }
        fprintf(stream, ".,.) = \n");

        _print_matrix(stream, *data_ptr, dims, linesize, fmt, 1);
        
        // 更新数据指针，移动到下一个2D切片
        *data_ptr += dims[0] * dims[1];
        return;
    }
    
    // 递归深入
    for (int i = 0; i < dims[0]; ++i) {
        counter[shape_get_ndim(counter) - ndim] = i; // This is also wrong
        _print_tensor_recursive(stream, data_ptr, dims + 1, ndim - 1, counter, linesize, fmt);
        if (i < dims[0] - 1) fprintf(stream, "\n");
    }
}

// 修正后的_print_tensor_recursive
static void _print_tensor_recursive_fixed(FILE* stream, double** data_ptr, int current_dim, const int* full_dims, int total_dims, int* counter, int linesize, const PrintFormat* fmt) {
    if (total_dims - current_dim == 2) {
        // Base case: we have a 2D slice to print.
        fprintf(stream, "(");
        for(int i = 0; i < total_dims - 2; ++i) {
            fprintf(stream, "%d,", counter[i] + 1);
        }
        fprintf(stream, ".,.) = \n");

        _print_matrix(stream, *data_ptr, &full_dims[current_dim], linesize, fmt, 1);
        
        // Advance the data pointer past this 2D slice
        *data_ptr += full_dims[current_dim] * full_dims[current_dim + 1];
        return;
    }

    // Recursive step
    for (int i = 0; i < full_dims[current_dim]; ++i) {
        counter[current_dim] = i;
        _print_tensor_recursive_fixed(stream, data_ptr, current_dim + 1, full_dims, total_dims, counter, linesize, fmt);
        if (i < full_dims[current_dim] - 1) {
            fprintf(stream, "\n");
        }
    }
}


// 内部打印主函数
static void _tensor_print_stream(FILE* stream, const Tensor t, int linesize) {
    if (t == NULL) {
        fprintf(stream, "[ Tensor (NULL) ]\n");
        return;
    }

    int ndim = shape_get_ndim(t->shape);
    const int* dims = shape_get_dims(t->shape);

    double* data_as_double = NULL;
    if (_tensor_to_double_array(t, &data_as_double) != 0) {
        return; // 转换失败
    }

    if (shape_get_elements_count(t->shape) == 0) {
         fprintf(stream, "[]\n");
    } else if (ndim == 0) {
        fprintf(stream, "%.4f\n", data_as_double[0]);
    } else if (ndim == 1) {
        PrintFormat fmt = _calculate_print_format(data_as_double, dims[0]);
        for (int i = 0; i < dims[0]; ++i) {
            _print_value(stream, data_as_double[i], &fmt);
            fprintf(stream, "\n");
        }
    } else if (ndim == 2) {
        PrintFormat fmt = _calculate_print_format(data_as_double, dims[0] * dims[1]);
        _print_matrix(stream, data_as_double, dims, linesize, &fmt, 0);
    } else { // ndim > 2
        size_t n_elements = shape_get_elements_count(t->shape);
        PrintFormat fmt = _calculate_print_format(data_as_double, n_elements);
        
        int* counter = (int*)safecalloc(ndim - 2, sizeof(int));
        if (!counter) { free(data_as_double); return; }

        double* data_ptr_copy = data_as_double;
        _print_tensor_recursive_fixed(stream, &data_ptr_copy, 0, dims, ndim, counter, linesize, &fmt);

        free(counter);
    }
    
    // 打印最后的摘要信息
    fprintf(stream, "[Tensor of shape: ");
    shape_print(t->shape);
    // 你可以添加 dtype 的打印
    // fprintf(stream, ", dtype: %s", _get_dtype_string(t->dtype));
    fprintf(stream, "]\n");

    free(data_as_double);
}


// --- 公开 API 实现 ---

void tensor_print_opts(const Tensor t, int linesize) {
    _tensor_print_stream(stdout, t, linesize);
}

void tensor_print(const Tensor t) {
    tensor_print_opts(t, 80); // 默认行宽为80
}