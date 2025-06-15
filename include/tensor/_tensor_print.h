#ifndef _TENSOR_PRINT_H
#define _TENSOR_PRINT_H

#include "tensor/_tensor_core.h" 

/**
 * @brief Prints a human-readable representation of the tensor to stdout.
 * This function intelligently formats the output for readability, similar
 * to how libraries like PyTorch and NumPy display tensors. It correctly
* handles tensors with non-contiguous memory layouts (e.g., views from
* permute operations).
 *
 * @param t The tensor to be printed.
 */
void tensor_print(const Tensor t);

#endif // _TENSOR_PRINT_H