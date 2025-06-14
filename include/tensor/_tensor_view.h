#ifndef _TENSOR_VIEW_H
#define _TENSOR_VIEW_H

#include "tensor/_tensor_core.h"
#include "tensor/_shape.h" 
#include <stdbool.h>

/**
 * @brief 创建一个具有新形状的张量视图。
 * * 新张量与原张量共享底层数据。因此，元素的总数必须保持不变。
 * 返回的张量是一个视图，它不拥有数据（owns_data=false）。
 * 使用完毕后，请务必对其调用 tensor_free()。
 *
 * @param t 要操作的原始张量。
 * @param new_shape 描述新形状的 Shape 对象。
 * @return 一个具有新形状的张量视图，如果失败则返回 NULL。
 */
Tensor tensor_reshape(const Tensor t, const Shape new_shape);

/**
 * @brief 置换张量的维度。
 * * 返回的张量是一个视图，与原张量共享数据。
 * 返回的张量是一个视图，它不拥有数据（owns_data=false）。
 * 使用完毕后，请务必对其调用 tensor_free()。
 *
 * @param t 要操作的原始张量。
 * @param axes 定义维度新顺序的数组。(注意：这个参数仍然是 int*，因为它代表的是轴的顺序，而不是一个形状)
 * @return 一个维度被置换后的张量视图，如果失败则返回 NULL。
 */
Tensor tensor_permute(const Tensor t, const int* axes);

/**
 * @brief 检查张量在内存中是否是行主序连续的。
 *
 * @param t 要检查的张量。
 * @return 如果张量是连续的，返回 true，否则返回 false。
 */
bool tensor_is_contiguous(const Tensor t);

/**
 * @brief 返回一个内存连续的张量。
 * * 如果输入的张量已经是连续的，将返回该张量的一个副本（copy）。
 * 如果不是，将创建一个新的连续张量，并复制数据。
 * 返回的张量总是拥有其数据（owns_data=true）。
 *
 * @param t 要操作的张量。
 * @return 一个新的连续张量。请记得使用 tensor_free() 释放它。
 */
Tensor tensor_contiguous(const Tensor t);

/**
 * @brief 通过广播将张量扩展到更大的尺寸。
 * * 返回的张量是一个视图，不拥有其数据。
 * 只有当原始维度大小为1时，该维度才能被扩展。
 *
 * @param t 要操作的原始张量。
 * @param target_shape 目标形状的 Shape 对象。
 * @return 一个具有扩展形状的张量视图，如果失败则返回 NULL。
 */
Tensor tensor_expand(const Tensor t, const Shape target_shape);


#endif // _TENSOR_VIEW_H