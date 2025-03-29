#include "../include/utils.h"
#include "../include/nn.h"
#include "../include/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// 错误处理
static void (*error_handler)(const char* msg) = NULL;

void set_error_handler(void (*handler)(const char* msg)) {
    error_handler = handler;
}

void report_error(const char* format, ...) {
    char msg[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(msg, sizeof(msg), format, args);
    va_end(args);
    
    if (error_handler) {
        error_handler(msg);
    } else {
        fprintf(stderr, "Error: %s\n", msg);
        exit(1);
    }
}

// 内存管理
void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        report_error("Failed to allocate memory");
    }
    return ptr;
}

void* safe_calloc(size_t count, size_t size) {
    void* ptr = calloc(count, size);
    if (!ptr) {
        report_error("Failed to allocate memory");
    }
    return ptr;
}

void* safe_realloc(void* ptr, size_t size) {
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr) {
        report_error("Failed to reallocate memory");
    }
    return new_ptr;
}

// 随机数生成
static unsigned int seed = 0;

void set_seed(unsigned int s) {
    seed = s;
    srand(seed);
}

float random_uniform(float min, float max) {
    return min + (max - min) * (float)rand() / RAND_MAX;
}

float random_normal(float mean, float std) {
    // Box-Muller变换
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return mean + std * z;
}

// 数据加载器
DataLoader* create_dataloader(Tensor* data, Tensor* labels, int batch_size) {
    if (!data || !labels || batch_size <= 0) {
        report_error("Invalid dataloader parameters");
        return NULL;
    }
    
    DataLoader* loader = safe_malloc(sizeof(DataLoader));
    loader->data = data;
    loader->labels = labels;
    loader->size = data->storage->size;
    loader->batch_size = batch_size;
    loader->current_idx = 0;
    
    return loader;
}

void dataloader_reset(DataLoader* loader) {
    if (!loader) return;
    loader->current_idx = 0;
}

int dataloader_next(DataLoader* loader, Tensor** batch_data, Tensor** batch_labels) {
    if (!loader || !batch_data || !batch_labels) return 0;
    
    int remaining = loader->size - loader->current_idx;
    int current_batch_size = (remaining < loader->batch_size) ? remaining : loader->batch_size;
    
    if (current_batch_size <= 0) return 0;
    
    // 创建批次数据
    int dims[1] = {current_batch_size};
    *batch_data = tensor_create(1, dims, NULL);
    *batch_labels = tensor_create(1, dims, NULL);
    
    // 复制数据
    memcpy((*batch_data)->storage->data,
           loader->data->storage->data + loader->current_idx,
           current_batch_size * sizeof(float));
    
    memcpy((*batch_labels)->storage->data,
           loader->labels->storage->data + loader->current_idx,
           current_batch_size * sizeof(float));
    
    loader->current_idx += current_batch_size;
    return current_batch_size;
}

void dataloader_free(DataLoader* loader) {
    if (!loader) return;
    free(loader);
}

// 模型保存和加载
void save_model(const char* filename, Module* model) {
    if (!filename || !model) {
        report_error("Invalid parameters for save_model");
        return;
    }
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        report_error("Failed to open file for writing: %s", filename);
        return;
    }
    
    // 保存模型类型
    fwrite(model->name, strlen(model->name) + 1, 1, file);
    
    // 保存参数
    for (int i = 0; i < model->num_parameters; i++) {
        Tensor* param = model->parameters[i];
        if (!param) continue;
        
        // 保存参数维度
        fwrite(&param->num_dims, sizeof(int), 1, file);
        fwrite(param->dims, sizeof(int), param->num_dims, file);
        
        // 保存参数数据
        fwrite(param->storage->data, sizeof(float), param->storage->size, file);
    }
    
    fclose(file);
}

Module* load_model(const char* filename) {
    if (!filename) {
        report_error("Invalid filename for load_model");
        return NULL;
    }
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        report_error("Failed to open file for reading: %s", filename);
        return NULL;
    }
    
    // 读取模型类型
    char model_type[32];
    fread(model_type, 1, sizeof(model_type), file);
    
    // 根据模型类型创建相应的模型
    Module* model = NULL;
    if (strcmp(model_type, "Linear") == 0) {
        // 读取线性层参数
        int in_features, out_features;
        fread(&in_features, sizeof(int), 1, file);
        fread(&out_features, sizeof(int), 1, file);
        
        model = (Module*)nn_linear(in_features, out_features, 1);
    }
    // 可以添加其他模型类型的加载逻辑
    
    if (!model) {
        fclose(file);
        report_error("Unsupported model type: %s", model_type);
        return NULL;
    }
    
    // 加载参数
    for (int i = 0; i < model->num_parameters; i++) {
        Tensor* param = model->parameters[i];
        if (!param) continue;
        
        // 读取参数维度
        int num_dims;
        fread(&num_dims, sizeof(int), 1, file);
        
        int* dims = safe_malloc(num_dims * sizeof(int));
        fread(dims, sizeof(int), num_dims, file);
        
        // 读取参数数据
        size_t size = 1;
        for (int j = 0; j < num_dims; j++) {
            size *= dims[j];
        }
        
        float* data = safe_malloc(size * sizeof(float));
        fread(data, sizeof(float), size, file);
        
        // 更新参数
        free(param->dims);
        param->dims = dims;
        param->num_dims = num_dims;
        
        free(param->storage->data);
        param->storage->data = data;
        param->storage->size = size;
    }
    
    fclose(file);
    return model;
}
