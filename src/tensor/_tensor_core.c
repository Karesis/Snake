#include "tensor/_tensor_core.h"

#include "utils/_malloc.h"
#include "tensor/_shape.h"

#include <stdint.h> // for int32_t
#include <stddef.h> // for size_t
#include <stdlib.h> // for free(), NULL
#include <string.h> // for memcpy()
#include <stdbool.h> // for bool, true, false

struct _tensor
{
    void* _data;
    Shape _shape;
    DataType _dtype;
    bool _owns_data;
};

static size_t _get_dtype_size(DataType dtype);

Tensor 
tensor_create(const Shape shape, DataType dtype)
{
    Tensor new = safemalloc(sizeof(struct _tensor));
    if (new == NULL) return NULL;

    
    size_t num_elements = shape_get_elements_count(shape);
    size_t dtype_size = _get_dtype_size(dtype);
    new->_data = safecalloc(num_elements, dtype_size);
    if (new->_data == NULL)
    {
        free(new);
        return NULL;
    }

    new->_shape = shape_copy(shape);
    if (new->_shape == NULL)
    {
        free(new->_data);
        free(new);
        return NULL;
    }

    new->_dtype = dtype;
    new->_owns_data = true;

    return new;
}

/**
 * @brief Creates a new tensor and initializes it with data from a provided buffer.
 */
Tensor
tensor_from_data(const void* data, const Shape shape, DataType dtype)
{
    // 1. 先创建一个具有正确尺寸的、零初始化的张量。
    //    这个步骤处理了所有的内存分配和结构体初始化。
    Tensor new_tensor = tensor_create(shape, dtype);
    if (new_tensor == NULL)
    {
        return NULL; // tensor_create 失败，直接返回
    }

    // 2. 如果外部数据源是有效的，则将数据复制到新张量中。
    if (data != NULL)
    {
        size_t num_elements = shape_get_elements_count(shape);
        size_t dtype_size = _get_dtype_size(dtype);
        size_t num_bytes = num_elements * dtype_size;
        
        // 将用户提供的数据，拷贝到新分配的内存中
        memcpy(new_tensor->_data, data, num_bytes);
    }

    return new_tensor;
}

void
tensor_free(Tensor tensor)
{
    if (tensor == NULL) return;

    if (tensor->_owns_data)
        free(tensor->_data);

    shape_free(tensor->_shape);
    free(tensor);
}

Tensor
tensor_copy(const Tensor other)
{
    if (other == NULL) return NULL;

    Tensor new = tensor_create(other->_shape, other->_dtype);
    if (new == NULL) return NULL;

    size_t num_elements = shape_get_elements_count(other->_shape);
    size_t dtype_size = _get_dtype_size(other->_dtype);
    size_t num_bytes = num_elements * dtype_size;
    memcpy(new->_data, other->_data, num_bytes);

    return new;
}

static size_t
_get_dtype_size(DataType dtype)
{
    switch (dtype)
    {
        case DTYPE_I32:
            return sizeof(int32_t);
            
        case DTYPE_F32:
            return sizeof(float);

        case DTYPE_F64:
            return sizeof(double);
        
        default:
            return 0;
    }
}

Shape 
tensor_get_shape(const Tensor tensor)
{
    return tensor->_shape;
}

void*
tensor_get_data(const Tensor tensor)
{
    return tensor->_data;
}

DataType
tensor_get_dtype(const Tensor tensor)
{
    return tensor->_dtype;
}

int
tensor_get_ndim(const Tensor tensor)
{
    // 将问题委托给 shape 对象
    if (tensor == NULL) return 0;
    return shape_get_ndim(tensor->_shape);
}

int
tensor_get_dim(const Tensor tensor, int axis)
{
    // 将问题委托给 shape 对象
    if (tensor == NULL) return -1;
    return shape_get_dim(tensor->_shape, axis);
}

const size_t*
tensor_get_strides(const Tensor tensor)
{
    // 将问题委托给 shape 对象
    if (tensor == NULL) return NULL;
    return shape_get_strides(tensor->_shape);
}

// 这个函数返回的就是一个逻辑上的elements_count.
// physical elements counts在库中统一不做要求和实现。
size_t
tensor_get_elements_count(const Tensor tensor)
{
    // 将问题委托给 shape 对象
    if (tensor == NULL) return 0;
    return shape_get_elements_count(tensor->_shape);
}

size_t
tensor_get_item_size(const Tensor tensor)
{
    // 这个问题取决于 dtype，所以调用内部的辅助函数
    if (tensor == NULL) return 0;
    return _get_dtype_size(tensor->_dtype);
}

/**
 * @brief (Internal) Creates a new Tensor view on existing data.
 */
Tensor
_tensor_create_view(void* data, const Shape new_shape, DataType dtype)
{
    // 1. 检查传入的参数是否有效
    if (data == NULL || new_shape == NULL)
        return NULL;

    // 2. 为 Tensor 结构体本身分配内存
    Tensor new_tensor = safemalloc(sizeof(struct _tensor));
    if (new_tensor == NULL)
        return NULL;

    // 3. 填充结构体字段
    new_tensor->_data = data;           // <-- 关键点1: 直接“借用”数据指针
    new_tensor->_dtype = dtype;         // 设置数据类型
    new_tensor->_owns_data = false;     // <-- 关键点2: 明确表示不拥有数据

    new_tensor->_shape = shape_copy(new_shape);
    if (new_tensor->_shape == NULL)
    {
        free(new_tensor);
        return NULL;
    }

    return new_tensor;
}

void*
tensor_get_element_ptr(const Tensor source_tensor, const int* coords)
{
    if (source_tensor == NULL || coords == NULL) return NULL;

    const int ndim = tensor_get_ndim(source_tensor);

    // TODO: 在生产级的库中，这里应该检查 coords 是否越界
    // for (int i = 0; i < ndim; i++) {
    //     if (coords[i] < 0 || coords[i] >= tensor_get_dim(t, i)) {
    //         return NULL; // Out of bounds
    //     }
    // }

    // --- 这就是通用的“计程车”逻辑 ---
    size_t offset_in_elements = 0;
    const size_t* strides = tensor_get_strides(source_tensor);
    for (int i = 0; i < ndim; i++) {
        offset_in_elements += coords[i] * strides[i];
    }
    // --- 逻辑结束 ---

    // 根据偏移量计算最终的物理地址
    const char* base_data = (const char*)tensor_get_data(source_tensor);
    const size_t item_size = tensor_get_item_size(source_tensor);
    
    void* element_ptr = (void*)(base_data + offset_in_elements * item_size);

    return element_ptr;
}