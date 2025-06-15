#include "tensor/_tensor_print.h"
#include "tensor/_tensor_core.h"
#include "tensor/_shape.h"
#include "utils/_malloc.h"

#include <stdio.h> // for fprintf(), stdout
#include <stdlib.h> // for free()
#include <string.h> // for memset()
#include <stdbool.h> // for bool, true, false
#include <math.h> // for isfinite(), floor(), fabs(), log10()

typedef enum 
{
    FORMAT_DEFAULT,
    FORMAT_SCIENTIFIC,
    FORMAT_FIXED
} 
FormatType;

typedef struct 
{
    int width;
    int precision;
    FormatType type;
} 
PrintFormat;

static PrintFormat 
_calculate_print_format(const double* data, size_t n) 
{
    if (n == 0) 
        return (PrintFormat){0, 4, FORMAT_DEFAULT};

    bool int_mode = true;
    for (size_t i = 0; i < n; ++i) 
    {
        if (isfinite(data[i]) && data[i] != floor(data[i])) 
        {
            int_mode = false;
            break;
        }
    }

    double exp_min = 0.0, exp_max = 0.0;
    bool first_finite = true;
    for (size_t i = 0; i < n; ++i) 
    {
        double z = fabs(data[i]);
        if (isfinite(z) && z > 0) 
        {
            if (first_finite) 
            {
                exp_min = exp_max = z;
                first_finite = false;
            } 
            else 
            {
                if (z < exp_min) exp_min = z;
                if (z > exp_max) exp_max = z;
            }
        }
    }

    if (!first_finite) // 如果找到了至少一个有限的非零数
    { 
        exp_min = floor(log10(exp_min));
        exp_max = floor(log10(exp_max));
    }

    if (int_mode) 
    {
        if (exp_max > 9) 
            return (PrintFormat){11, 4, FORMAT_SCIENTIFIC};
        else 
        {
            return (PrintFormat){(int)exp_max + 2, 0, FORMAT_DEFAULT};
        }
    } 
    else 
    {
        if (exp_max - exp_min > 4) 
            return (PrintFormat){11, 4, FORMAT_SCIENTIFIC};
        else 
        {
            int precision = 4;
            int width = (exp_max > 0 ? (int)exp_max : 0) + precision + 2;
            return (PrintFormat){width, precision, FORMAT_FIXED};
        }
    }
}

static void 
_print_value(FILE* stream, double value, const PrintFormat* fmt)
{
    switch (fmt->type) 
    {
        case FORMAT_DEFAULT:
            fprintf(stream, "%*g", fmt->width, value);
            break;

        case FORMAT_SCIENTIFIC:
            fprintf(stream, "%*.*e", fmt->width, fmt->precision, value);
            break;

        case FORMAT_FIXED:
            fprintf(stream, "%*.*f", fmt->width, fmt->precision, value);
            break;
    }
}

static void 
_print_recursive(FILE* stream, const Tensor t, int* coords, int current_dim, const PrintFormat* fmt) 
{
    const int ndim = tensor_get_ndim(t);
    const int size_this_dim = tensor_get_dim(t, current_dim);

    fprintf(stream, "[");

    if (current_dim == ndim - 1) 
    { // Base case: a single row
        for (int i = 0; i < size_this_dim; i ++) 
        {
            coords[current_dim] = i;
            void* elem_ptr = tensor_get_element_ptr(t, coords);
            
            double val = 0.0;
            switch (tensor_get_dtype(t)) 
            {
                case DTYPE_F32: val = (double)(*(float*)elem_ptr); break;
                case DTYPE_F64: val = *(double*)elem_ptr; break;
                case DTYPE_I32: val = (double)(*(int*)elem_ptr); break;
            }

            _print_value(stream, val, fmt);

            if (i < size_this_dim - 1) 
            {
                fprintf(stream, ", ");
            }
        }
    } 
    else // Recursive step
    { 
        for (int i = 0; i < size_this_dim; i++) 
        {
            coords[current_dim] = i;
            if (i > 0) 
            {
                fprintf(stream, ",\n%*s", (current_dim + 1), "");
            }
            _print_recursive(stream, t, coords, current_dim + 1, fmt);
        }
    }
    fprintf(stream, "]");
}

static void 
_tensor_print_stream(FILE* stream, const Tensor t) 
{
    if (t == NULL) 
    {
        fprintf(stream, "[ Tensor (NULL) ]\n");
        return;
    }

    const int ndim = tensor_get_ndim(t);
    const size_t num_elements = tensor_get_elements_count(t);

    if (num_elements == 0) 
    {
        fprintf(stream, "[]\n");
    }
    else if (ndim == 0) 
    {
        void* elem_ptr = tensor_get_data(t);
        double val = (tensor_get_dtype(t) == DTYPE_F32) ? (double)(*(float*)elem_ptr) : (double)(*(int*)elem_ptr); // 简化处理
        fprintf(stream, "%.4f\n", val);
    } 
    else 
    {
        // --- 阶段一：预扫描，获取所有元素的值用于格式计算 ---
        double* temp_data = (double*)safemalloc(num_elements * sizeof(double));
        if (!temp_data) return;

        int* coords = (int*)safecalloc(ndim, sizeof(int));
        if (!coords) { free(temp_data); return; }

        for (size_t i = 0; i < num_elements; i++) 
        {
            // "里程表"逻辑，用于遍历所有逻辑坐标
            void* elem_ptr = tensor_get_element_ptr(t, coords);
            switch (tensor_get_dtype(t)) {
                case DTYPE_F32: temp_data[i] = (double)(*(float*)elem_ptr); break;
                case DTYPE_F64: temp_data[i] = *(double*)elem_ptr; break;
                case DTYPE_I32: temp_data[i] = (double)(*(int*)elem_ptr); break;
            }

            // 更新里程表
            int current_d = ndim - 1;
            while (current_d >= 0) {
                coords[current_d]++;
                if (coords[current_d] < tensor_get_dim(t, current_d)) break;
                coords[current_d] = 0;
                current_d--;
            }
        }
        
        // 计算最佳格式
        PrintFormat fmt = _calculate_print_format(temp_data, num_elements);
        free(temp_data); // 临时数据用完即焚

        // --- 阶段二：使用计算出的格式进行递归打印 ---
        memset(coords, 0, ndim * sizeof(int)); // 重置坐标
        _print_recursive(stream, t, coords, 0, &fmt);
        fprintf(stream, "\n");
        
        free(coords);
    }
    
    // 打印最后的摘要信息
    fprintf(stream, "[Tensor of shape: ");
    shape_print(tensor_get_shape(t));
    fprintf(stream, "]\n");
}

// --- 公开 API 实现 ---
void tensor_print(const Tensor t) 
{
    _tensor_print_stream(stdout, t);
}