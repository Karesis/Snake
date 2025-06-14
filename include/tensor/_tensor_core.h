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


Tensor tensor_create(const Shape shape, DataType dtype);
void tensor_free(Tensor tensor);
Tensor tensor_copy(const Tensor tensor);
/**
 * @brief Creates a new tensor and initializes it with data from a provided buffer.
 * The function creates a new tensor that OWNS its data. It allocates new memory
 * and copies the content from the `data` buffer.
 *
 * @param data A const pointer to the buffer containing the data to copy.
 * @param shape The shape of the tensor to be created.
 * @param dtype The data type of the elements in the buffer.
 * @return A new Tensor object on success, or NULL on failure.
 */
Tensor tensor_from_data(const void* data, const Shape shape, DataType dtype);


Shape tensor_get_shape(const Tensor tensor);
void* tensor_get_data(const Tensor tensor);
DataType tensor_get_dtype(const Tensor tensor);
int tensor_get_ndim(const Tensor tensor);
int tensor_get_dim(const Tensor tensor, int axis);
const size_t* tensor_get_strides(const Tensor tensor);
size_t tensor_get_elements_count(const Tensor tensor);
size_t tensor_get_item_size(const Tensor tensor);

/**
 * @brief (Internal) Creates a new Tensor that is a "view" on existing data.
 * This new tensor does NOT own the data. The caller is responsible for providing
 * a new Shape object, which the created view WILL take ownership of.
 *
 * @param data A pointer to the existing data to be shared.
 * @param shape The new shape for the view. The view takes ownership of this shape object.
 * @param dtype The data type of the elements.
 * @return A new Tensor view, or NULL on failure.
 */
Tensor tensor_create_view(void* data, Shape shape, DataType dtype);

/**
 * @brief Gets a pointer to the element at the specified logical coordinates.
 * WARNING: This can be slow if used in a tight loop. It's intended for
 * debugging or accessing/modifying a small number of specific elements.
 *
 * @param t The tensor to access.
 * @param coords An array of integers representing the logical coordinates.
 * The length of this array must be equal to the tensor's ndim.
 * @return A void pointer to the element's data on success, or NULL on failure
 * (e.g., if coords are out of bounds). The user should cast this
 * pointer to the appropriate type (e.g., float*).
 */
void* tensor_get_element_ptr(const Tensor t, const int* coords);

#endif // _TENSOR_CORE_H