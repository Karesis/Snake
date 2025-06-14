#include "tensor/_tensor_view.h"
#include <stdio.h>

Tensor
tensor_reshape(const Tensor tensor, const Shape new_shape)
{
    // --- 1. 获取旧形状信息 ---
    const Shape old_shape = tensor_get_shape(tensor);

    // --- 2. 验证 ---
    // a. 检查元素总数是否匹配
    if (shape_get_elements_count(old_shape) != shape_get_elements_count(new_shape))
    {
        fprintf(stderr, "Error: cannot reshape, total number of elements must remain the same.\n");
        return NULL;
    }

    // b. Reshape 操作要求原始张量是连续的
    if (!tensor_is_contiguous(tensor))
    {
        fprintf(stderr, "Error: tensor_reshape requires a contiguous tensor. Call tensor_contiguous() first.\n");
        return NULL;
    }

    // 创建一个新的 Tensor 结构体，但共享数据
    // 你可以在 _tensor_core.c 中创建一个内部函数来完成这个操作
    Tensor view = _tensor_create_view
    (
        tensor_get_data(tensor), 
        new_shape, 
        tensor_get_dtype(tensor)
    );
    
    return view;
}

bool
tensor_is_contiguous(const Tensor tensor)
{
    return shape_is_contiguous(tensor_get_shape(tensor));
}

/**
 * @brief 置换张量的维度。
 */
Tensor
tensor_permute(const Tensor t, const int* axes)
{
    if (t == NULL) return NULL;

    // 1. 获取原始形状
    const Shape old_shape = tensor_get_shape(t);

    // 2. 调用 shape 模块计算置换后的新形状
    //    所有复杂的验证和计算都在 shape_permute 中完成
    Shape new_permuted_shape = shape_permute(old_shape, axes);
    if (new_permuted_shape == NULL) {
        // shape_permute 内部会打印错误信息
        return NULL;
    }

    // 3. 使用视图工厂函数创建视图
    //    新张量借用旧张量的数据，但拥有一个新的 Shape 对象
    Tensor view = _tensor_create_view
    (
        tensor_get_data(t),
        new_permuted_shape,
        tensor_get_dtype(t)
    );

    return view;
}

/**
 * @brief 通过广播将张量扩展到更大的尺寸。
 */
Tensor
tensor_expand(const Tensor t, const Shape target_shape)
{
    if (t == NULL || target_shape == NULL) return NULL;

    // 1. 获取原始形状
    const Shape source_shape = tensor_get_shape(t);

    // 2. 调用 shape 模块计算扩展后的新形状
    //    所有复杂的验证和 stride 计算都在 shape_expand 中完成
    Shape new_expanded_shape = shape_expand(source_shape, target_shape);
    if (new_expanded_shape == NULL) {
        // shape_expand 内部会打印错误信息
        return NULL;
    }

    // 3. 使用视图工厂函数创建视图
    Tensor view = _tensor_create_view
    (
        tensor_get_data(t),
        new_expanded_shape,
        tensor_get_dtype(t)
    );

    return view;
}

/**
 * @brief 返回一个内存连续的张量。
 */
Tensor
tensor_contiguous(const Tensor tensor)
{
    if (tensor == NULL) return NULL;

    // --- 情况1: 张量已经是连续的 ---
    if (tensor_is_contiguous(tensor)) {
        // 直接返回一个深拷贝，新的张量拥有自己的数据
        return tensor_copy(tensor);
    }

    // --- 情况2: 张量不连续 ---
    const Shape shape = tensor_get_shape(tensor);
    const DataType dtype = tensor_get_dtype(tensor);

    // a. 创建一个新的、内存连续的目标张量
    Tensor contiguous_tensor = tensor_create(shape, dtype);
    if (contiguous_tensor == NULL) {
        return NULL;
    }

    // b. 逐个元素地将数据从非连续的源(tensor)复制到连续的目标(contiguous_tensor)
    const int ndim = tensor_get_ndim(tensor);
    const size_t* strides = tensor_get_strides(tensor);
    const size_t item_size = tensor_get_item_size(tensor);
    const size_t num_elements = tensor_get_elements_count(tensor);

    // 获取裸指针
    const char* source_data = (const char*)tensor_get_data(tensor);
    char* dest_data = (char*)tensor_get_data(contiguous_tensor);

    // 用一个数组来像里程表一样追踪逻辑坐标
    int* coords = safecalloc(ndim, sizeof(int));
    if (coords == NULL) {
        tensor_free(contiguous_tensor);
        return NULL;
    }

    for (size_t i = 0; i < num_elements; i++) {
        // c. 根据坐标和步长，计算源数据在内存中的物理偏移量
        size_t source_offset = 0;
        for (int d = 0; d < ndim; d++) {
            source_offset += coords[d] * strides[d];
        }

        // d. 复制单个元素的数据
        //    源地址：基地址 + 物理偏移量 * 单个元素大小
        //    目标地址：基地址 + 线性索引 * 单个元素大小
        memcpy(dest_data + i * item_size, 
               source_data + source_offset * item_size,
               item_size);
        
        // e. 更新里程表（逻辑坐标）
        int current_dim = ndim - 1;
        while (current_dim >= 0) {
            coords[current_dim]++;
            if (coords[current_dim] < tensor_get_dim(tensor, current_dim)) {
                break; // 当前维度未溢出，停止进位
            }
            coords[current_dim] = 0; // 当前维度溢出，归零并向前一位进位
            current_dim--;
        }
    }
    
    free(coords);
    return contiguous_tensor;
}