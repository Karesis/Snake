#include "tensor/_tensor_core.h"
#include "utils/_malloc.h"
#include "tensor/_shape.h"

struct _tensor
{
    void* data;
    Shape shape;
    DataType dtype;
    bool owns_data;
};

Tensor 
tensor_create(const Shape shape, DataType dtype)
{
    Tensor new = safemalloc(sizeof(struct _tensor));
    if (new == NULL) return NULL;

    
    size_t num_elements = shape_get_elements_count(shape);
    size_t dtype_size = _get_dtype_size(dtype);
    new->data = safecalloc(num_elements, dtype_size);
    if (new->data == NULL)
    {
        free(new);
        return NULL;
    }

    new->shape = shape_copy(shape);
    if (new->shape == NULL)
    {
        free(new->data);
        free(new);
        return NULL;
    }

    new->dtype = dtype;
    new->owns_data = true;

    return new;
}

void
tensor_free(Tensor tensor)
{
    if (tensor == NULL) return;

    if (tensor->owns_data)
        free(tensor->data);

    shape_free(tensor->shape);
    free(tensor);
}

Tensor
tensor_copy(const Tensor other)
{
    if (other == NULL) return NULL;

    Tensor new = tensor_create(other->shape, other->dtype);
    if (new == NULL) return NULL;

    size_t num_elements = shape_get_elements_count(other->shape);
    size_t dtype_size = _get_dtype_size(other->dtype);
    size_t num_bytes = num_elements * dtype_size;
    memcpy(new->data, other->data, num_bytes);

    return new;
}