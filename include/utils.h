#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdarg.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 前向声明
struct Tensor;
struct Module;
typedef struct Tensor Tensor;
typedef struct Module Module;

// 内存管理
void* safe_malloc(size_t size);
void* safe_calloc(size_t count, size_t size);
void* safe_realloc(void* ptr, size_t size);

// 模型保存和加载
void save_model(const char* filename, Module* model);
Module* load_model(const char* filename);

// 数据加载和处理
typedef struct {
    Tensor* data;
    Tensor* labels;
    int size;
    int batch_size;
    int current_idx;
} DataLoader;

DataLoader* create_dataloader(Tensor* data, Tensor* labels, int batch_size);
void dataloader_reset(DataLoader* loader);
int dataloader_next(DataLoader* loader, Tensor** batch_data, Tensor** batch_labels);
void dataloader_free(DataLoader* loader);

// 随机数生成
void set_seed(unsigned int seed);
float random_uniform(float min, float max);
float random_normal(float mean, float std);

// 错误处理
void set_error_handler(void (*handler)(const char* msg));
void report_error(const char* format, ...);

#endif // UTILS_H
