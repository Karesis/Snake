#ifndef _SHAPE_H
#define _SHAPE_H

#include <stdbool.h>
#include <stddef.h> // For size_t

// --- Opaque Pointer Definition ---
struct _shape;
typedef struct _shape* Shape;


// --- Lifecycle Functions ---

/**
 * @brief Creates a new Shape object from a given dimensions array.
 * @param dims An array of integers describing the dimensions.
 * @param ndim The number of dimensions.
 * @return A new Shape object on success, or NULL if memory allocation fails.
 */
Shape shape_create(const int* dims, int ndim);

/**
 * @brief Creates a copy of an existing Shape object.
 * @param other The Shape object to copy.
 * @return A new copy of the Shape object on success, or NULL on failure.
 */
Shape shape_copy(const Shape other);

/**
 * @brief Frees all memory associated with a Shape object.
 * @param shape The Shape object to free.
 */
void shape_free(Shape shape);


// --- Accessor Functions ---

/**
 * @brief Gets the number of dimensions of the shape.
 * @param shape The shape object.
 * @return The number of dimensions (ndim). Returns 0 if shape is NULL.
 */
int shape_get_ndim(const Shape shape);

/**
 * @brief Gets a const pointer to the dimensions array.
 * @param shape The shape object.
 * @return A const pointer to the internal dimensions array. The user must not modify its contents.
 */
const int* shape_get_dims(const Shape shape);

/**
 * @brief Gets the size of a specific dimension (axis).
 * @param shape The shape object.
 * @param axis The index of the axis to query (must be 0 <= axis < ndim).
 * @return The size of the dimension at the given axis. Returns -1 if shape is NULL or axis is out of bounds.
 */
int shape_get_dim(const Shape shape, int axis);

/**
 * @brief Gets a const pointer to the stride array. (For advanced use).
 * @param shape The shape object.
 * @return A const pointer to the internal stride array.
 */
const size_t* shape_get_strides(const Shape shape);


// --- Utility Functions ---

/**
 * @brief Gets the total number of elements in the shape.
 * @param shape The shape object.
 * @return The total number of elements. Returns 1 for a 0-dimension shape (scalar) and 0 for a NULL shape.
 */
size_t shape_get_elements_count(const Shape shape);

/**
 * @brief Checks if two shapes are equal by comparing their dimensions.
 * @param a The first Shape object.
 * @param b The second Shape object.
 * @return true if all dimensions match, false otherwise.
 */
bool shape_equals(const Shape a, const Shape b);

/**
 * @brief Prints the shape's contents to stdout, e.g., "Shape[3, 4, 5]".
 * @param shape The Shape object to print.
 */
void shape_print(const Shape shape);

/**
 * @brief Check if this shape explains a contiguaus data
 * @param shape is the Shape to be check.
 */
bool shape_is_contiguous(const Shape shape);

Shape shape_permute(const Shape source_shape, const int* axes);

Shape shape_expand(const Shape source_shape, const Shape target_shape);

#endif // _SHAPE_H