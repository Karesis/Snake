#include "tensor/_shape.h"

#include "utils/_malloc.h"

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>



struct _shape
{
    int* _dims;
    size_t* _stride;
    int _ndim;
};

// --- Lifecycle Functions ---

/**
 * @brief Creates a new Shape object from a given dimensions array.
 */
Shape
shape_create(const int* dims, int ndim)
{
    Shape new = safemalloc(sizeof(struct _shape));
    if (new == NULL) return NULL;

    new->_dims = safemalloc(sizeof(int) * ndim);
    if (new->_dims == NULL)
    {
        free(new);
        return NULL;
    }
    memcpy(new->_dims, dims, sizeof(int) * ndim);
    new->_ndim = ndim;

    new->_stride = safemalloc(sizeof(size_t) * ndim);
    if (new->_stride == NULL)
    {
        free(new->_dims);
        free(new);
        return NULL;
    }

    // Calculate strides for row-major layout
    if (ndim > 0)
    {
        new->_stride[ndim-1] = 1;
        for (int i = ndim-2; i >= 0; i--)
            new->_stride[i] = new->_stride[i+1] * new->_dims[i+1];
    }

    return new;
}

/**
 * @brief Creates a copy of an existing Shape object.
 */
Shape
shape_copy(const Shape other)
{
    if (other == NULL) return NULL;

    Shape new = safemalloc(sizeof(struct _shape));
    if (new == NULL) return NULL;

    new->_ndim = other->_ndim;

    // Copy dimensions
    new->_dims = safemalloc(sizeof(int) * new->_ndim);
    if (new->_dims == NULL)
    {
        free(new);
        return NULL;
    }
    memcpy(new->_dims, other->_dims, sizeof(int) * new->_ndim);

    // Copy strides directly, no need to recalculate
    new->_stride = safemalloc(sizeof(size_t) * new->_ndim);
    if (new->_stride == NULL)
    {
        free(new->_dims);
        free(new);
        return NULL;
    }
    memcpy(new->_stride, other->_stride, sizeof(size_t) * new->_ndim);

    return new;
}

/**
 * @brief Frees all memory associated with a Shape object.
 */
void
shape_free(Shape shape)
{
    if (shape == NULL) return;

    free(shape->_dims);
    free(shape->_stride);
    free(shape);
}


// --- Accessor Functions ---

/**
 * @brief Gets the number of dimensions of the shape.
 */
int
shape_get_ndim(const Shape shape)
{
    if (shape == NULL)
        return 0;

    return shape->_ndim;
}

/**
 * @brief Gets a const pointer to the dimensions array.
 */
const int*
shape_get_dims(const Shape shape)
{
    if (shape == NULL)
        return NULL;

    return shape->_dims;
}

/**
 * @brief Gets the size of a specific dimension (axis).
 */
int
shape_get_dim(const Shape shape, int axis)
{
    if (shape == NULL)
        return -1; // Error: null shape

    if (axis < 0 || axis >= shape->_ndim)
        return -1; // Error: axis out of bounds

    return shape->_dims[axis];
}

/**
 * @brief Gets a const pointer to the stride array. (For advanced use).
 */
const size_t*
shape_get_strides(const Shape shape)
{
    if (shape == NULL)
        return NULL;

    return shape->_stride;
}


// --- Utility Functions ---

/**
 * @brief Gets the total number of elements in the shape.
 */
size_t
shape_get_elements_count(const Shape shape)
{
    if (shape == NULL)
        return 0;

    if (shape->_ndim == 0)
        return 1; // A scalar has one element

    size_t elements_count = 1;
    for (int i = 0; i < shape->_ndim; i++)
        elements_count *= shape->_dims[i];

    return elements_count;
}

/**
 * @brief Checks if two shapes are equal by comparing their dimensions.
 */
bool
shape_equals(const Shape a, const Shape b)
{
    // If they point to the same object, they are equal.
    if (a == b) return true;

    // If one is NULL and the other is not, they are not equal.
    if (a == NULL || b == NULL)
        return false;

    if (a->_ndim != b->_ndim)
        return false;

    // Check if all dimensions match.
    for (int i = 0; i < a->_ndim; i++)
        if (a->_dims[i] != b->_dims[i])
            return false;

    return true;
}

/**
 * @brief Prints the shape's contents to stdout, e.g., "Shape[3, 4, 5]".
 */
void
shape_print(const Shape shape)
{
    if (shape == NULL || shape->_dims == NULL || shape->_ndim == 0)
    {
        printf("Shape[]");
        return;
    }

    printf("Shape[");
    for (int i = 0; i < shape->_ndim; i++)
    {
        printf("%d", shape->_dims[i]);
        if (i < shape->_ndim - 1)
            printf(", ");
    }
    printf("]");
}

bool
shape_is_contiguous(const Shape shape)
{
    if (shape == NULL) return false;
    if (shape->_ndim == 0) return true;

    size_t expected_stride = 1;
    for (int i = shape->_ndim - 1; i >= 0; i --)
    {
        if (shape->_dims[i] != 1)
        {
            if (shape->_stride[i] != expected_stride)
                return false;
        }
        
        expected_stride *= shape->_dims[i];
    }

    return true;
}

Shape
shape_permute(const Shape source_shape, const int* axes)
{
    if (source_shape == NULL || axes == NULL) return NULL;

    int ndim = source_shape->_ndim;

    bool* axis_seen = safecalloc(ndim, sizeof(bool)); // 创建一个“清单”
    if (axis_seen == NULL) return NULL; // 内存分配失败

    for (int i = 0; i < ndim; i++) {
        int original_axis = axes[i];
        
        // 1. 检查范围
        if (original_axis < 0 || original_axis >= ndim) {
            fprintf(stderr, "Error: axis %d is out of bounds for tensor of dimension %d\n", original_axis, ndim);
            free(axis_seen);
            return NULL;
        }
        
        // 2. 检查唯一性
        if (axis_seen[original_axis]) {
            fprintf(stderr, "Error: duplicate axis %d found in axes array\n", original_axis);
            free(axis_seen);
            return NULL;
        }
        
        axis_seen[original_axis] = true; // 在清单上打勾
    }
    free(axis_seen); // 检查完毕，释放清单

    Shape new_shape = safemalloc(sizeof(struct _shape));
    if (new_shape == NULL) return NULL;

    new_shape->_ndim = ndim;

    // 分配新维度和新步长的内存
    new_shape->_dims = safemalloc(sizeof(int) * ndim);
    new_shape->_stride = safemalloc(sizeof(size_t) * ndim);
    
    if (new_shape->_dims == NULL || new_shape->_stride == NULL) {
        free(new_shape->_dims);
        free(new_shape->_stride);
        free(new_shape);
        return NULL;
    }

    // 根据 axes 重新排序 shape 和 stride
    for (int i = 0; i < ndim; i++)
    {
        int original_axis = axes[i];
        new_shape->_dims[i] = source_shape->_dims[original_axis];
        new_shape->_stride[i] = source_shape->_stride[original_axis];
    }

    return new_shape;

}

// broad rule is from the right to left. e.g:
// target: (5, 3, 4)
// source:    (3, 4)  <-- 右对齐
Shape
shape_expand(const Shape source_shape, const Shape target_shape)
{
    if (source_shape == NULL || target_shape == NULL) return NULL;

    const int source_ndim = shape_get_ndim(source_shape);
    const int* source_dims = shape_get_dims(source_shape);
    const size_t* source_strides = shape_get_strides(source_shape);

    const int target_ndim = shape_get_ndim(target_shape);
    const int* target_dims = shape_get_dims(target_shape);

    // --- 1. 验证兼容性 ---
    // a. 原始维度不能比目标维度多
    if (source_ndim > target_ndim) {
        fprintf(stderr, "Error: cannot expand to a shape with fewer dimensions.\n");
        return NULL;
    }

    // b. 从右向左逐一比较维度
    for (int i = 1; i <= source_ndim; i++) {
        int original_dim = source_dims[source_ndim - i];
        int target_dim = target_dims[target_ndim - i];
        
        // 维度的尺寸必须相等，或者是原始维度为1
        if (original_dim != target_dim && original_dim != 1) {
            fprintf(stderr, "Error: Incompatible shapes for expansion. A dimension must be 1 to be expanded.\n");
            return NULL;
        }
    }

    // --- 2. 创建并填充新的 Shape 对象 ---
    Shape new_shape = safemalloc(sizeof(struct _shape));
    if (new_shape == NULL) return NULL;
    
    new_shape->_ndim = target_ndim;

    // a. 新的 dims 就是 target_dims 的一个副本
    new_shape->_dims = safemalloc(sizeof(int) * target_ndim);
    if (new_shape->_dims == NULL) { free(new_shape); return NULL; }
    memcpy(new_shape->_dims, target_dims, sizeof(int) * target_ndim);

    // b. 计算新的 strides，这是广播的核心
    new_shape->_stride = safemalloc(sizeof(size_t) * target_ndim);
    if (new_shape->_stride == NULL) { free(new_shape->_dims); free(new_shape); return NULL; }

    int shape_diff = target_ndim - source_ndim;
    for (int i = 0; i < target_ndim; i++)
    {
        int source_dim_idx = i - shape_diff;

        // 如果这是一个新增的维度(在左侧)，或者原始维度是1，那么这就是一个广播维度
        if (source_dim_idx < 0 || source_dims[source_dim_idx] == 1) {
            new_shape->_stride[i] = 0; // <-- 广播的魔法！
        } else {
            // 否则，保持原来的步长
            new_shape->_stride[i] = source_strides[source_dim_idx];
        }
    }

    return new_shape;
}