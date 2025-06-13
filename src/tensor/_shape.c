#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

#include "tensor/_shape.h"
#include "utils/_malloc.h"

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