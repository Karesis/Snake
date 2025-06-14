#ifndef _TENSOR_CORE_H
#define _TENSOR_CORE_H

#include "tensor/_shape.h" 

struct _tensor;
typedef struct _tensor* Tensor;

typedef enum
{
    DTYPE_I32,
    DTYPE_F32,
    DTYPE_F64
} 
DataType;

// --- Core Lifecycle Functions ---

Tensor* tensor_create(const Shape shape, DataType dtype);
void tensor_free(Tensor* t);
Tensor* tensor_copy(const Tensor* t);
void tensor_print(const Tensor* t);


#endif // _TENSOR_CORE_H