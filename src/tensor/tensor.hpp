#pragma once //编译器指令（非标准但广泛支持），表示该头文件只会被包含一次，避免重复定义
#include "../core/llaisys_core.hpp" // 包含另一个头文件：引入核心功能。语法：#include "路径" 或 <路径>，这里使用相对路径。

#include <vector>
namespace llaisys { // 命名空间声明：定义llaisys命名空间，所有后续声明/定义都在此作用域内
    //命名空间块（{ ... }）内部可以包含几乎所有C++元素，包括变量声明、函数声明/定义、类/结构体定义、枚举、类型别名等。这是因为命名空间本质上是一个作用域（scope），它将内部的所有元素“限定”在该命名空间下。函数体（即函数的实现代码）可以直接放在命名空间内，这表示该函数属于该命名空间。
class Tensor;
using tensor_t = std::shared_ptr<Tensor>; //智能指针：共享所有权，自动引用计数释放。tensor_t 是 std::shared_ptr<Tensor> 的别名（比 typedef 更现代）。

struct TensorMeta {
    llaisysDataType_t dtype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides; // ptrdiff_t是带符号的指针差类型
};

class Tensor {
private:
    TensorMeta _meta;
    core::storage_t _storage;
    size_t _offset;
    Tensor(TensorMeta meta, core::storage_t storage, size_t offset = 0);

public:
    static tensor_t create(
        const std::vector<size_t> &shape,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU,
        int device = 0);
    ~Tensor() = default; // 析构函数：默认实现。语法：~类名() = default; 表示编译器生成默认析构。
    // Info
    std::byte *data();
    const std::byte *data() const;
    size_t ndim() const;
    const std::vector<size_t> &shape() const;
    const std::vector<ptrdiff_t> &strides() const;
    llaisysDataType_t dtype() const;
    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    size_t numel() const;
    size_t elementSize() const;

    std::string info() const;
    void debug() const;

    bool isContiguous() const;

    // Meta Transform
    tensor_t permute(const std::vector<size_t> &order) const;
    tensor_t slice(size_t dim, size_t start, size_t end) const;
    tensor_t view(const std::vector<size_t> &shape) const;

    // Load data from host memory
    void load(const void *src);

    // Challenging features
    tensor_t contiguous() const;
    tensor_t reshape(const std::vector<size_t> &shape) const;
    tensor_t to(llaisysDeviceType_t device_type, int device = -1) const;
};

} // namespace llaisys
