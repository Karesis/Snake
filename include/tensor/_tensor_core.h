#ifndef _TENSOR_CORE_H
#define _TENSOR_CORE_H

#include <stdbool.h>
#include "tensor/_shape.h" 

struct _tensor;
typedef struct _tensor* Tensor;

typedef enum
{
    DTYPE_F32,
    DTYPE_I32
} 
DataType;

// --- Core Lifecycle Functions ---

Tensor* tensor_create(const Shape shape, DataType dtype);
void tensor_free(Tensor* t);
Tensor* tensor_copy(const Tensor* t);
void tensor_print(const Tensor* t);


#endif // _TENSOR_CORE_H