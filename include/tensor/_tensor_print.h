#ifndef _TENSOR_PRINT_H
#define _TENSOR_PRINT_H

#include "tensor/_tensor_core.h" 

/**
 * @brief Prints a representation of the tensor to the standard output,
 * mimicking the style of PyTorch.
 *
 * @param t The tensor to print.
 * @param linesize The approximate maximum number of characters per line.
 */
void tensor_print_opts(const Tensor t, int linesize);

/**
 * @brief Prints a tensor with default line size (80).
 * @param t The tensor to print.
 */
void tensor_print(const Tensor t);

#endif // _TENSOR_PRINT_H