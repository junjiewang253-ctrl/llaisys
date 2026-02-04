#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring> // 内存操作库，如memcpy。
#include <numeric> // 数值算法，如accumulate。
#include <sstream> // 字符串流。
#include <algorithm>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {} // std::move用于移动语义，避免拷贝。


tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    // 这个函数返回一个 std::byte *指针，指向Tensor数据的起始地址（加上偏移量）
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    // std::accumulate：算法函数，计算乘积。语法：std::accumulate(迭代器开始, 结束, 初始值, 操作)
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    // 流操作：<< 用于插入字符串和变量。
    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    // 取出形状与步长（通常shape/strides的长度应一致）
    const auto &shape = this->shape();
    const auto &strides = this->strides();
    // 形状与步长度数不一致，直接判定为不连续
    if (shape.size() != strides.size()) {
        return false;
    }
    // 0维张量按惯例认为是连续的
    if (shape.empty()) {
        return true;
    }
    // 按C-order（行优先）连续布局检查：
    // 最后一维期望stride=1；往前每一维期望stride *= 下一维的size
    // 重要特例: size==1的维度在内存中不"展开"，stride可以是任意值，不影响连续性
    int64_t expected = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        const int64_t dim = static_cast<int64_t>(shape[i]);
        const int64_t st = static_cast<int64_t>(strides[i]);
        // 如果某一维为0，说明是空张量；通常也认为他是连续的（没有实际数据需要布局）
        if (dim == 0) {
            return true;
        }
        // 该维长度为1：无论stride是多少，都不影响数据在内存中的连续性
        if (dim == 1) {
            continue;
        }
        // 对于长度>1的维度，stride必须严格等于期望值
        if (st != expected) {
            return false;
        }
        // 更新下一轮（更高一维）期望的stride
        expected *= dim;
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    const size_t n = this->ndim(); // this指向当前调用该成员函数的对象
    // 检查order长度
    if (order.size() != n) {
        throw std::invalid_argument("Tensor::permute: order size must equal ndim");
    }
    // 检查order是一个[0...n-1]的排列（无越界、无重复）
    std::vector<bool> seen(n, false);
    for (size_t i = 0; i < n; ++i) {
        const size_t d = order[i];
        if (d >= n) {
            throw std::invalid_argument("Tensor::permute: order contains out-of-range dim");
        }
        if (seen[d]) {
            throw std::invalid_argument("Tensor::permute: order contains duplicate dims");
        }
        seen[d] = true;
    }
    // 构造新meta: shape/strides按order重排
    TensorMeta meta;
    meta.dtype = this->dtype();
    meta.shape.resize(n);
    meta.strides.resize(n);
    const auto &old_shape = this->shape();
    const auto &old_strides = this->strides();
    for (size_t i = 0; i < n; ++i) {
        const size_t src_dim = order[i];
        meta.shape[i] = old_shape[src_dim];
        meta.strides[i] = old_strides[src_dim];
    }
    // 共享storage + offset,不移动数据
    return std::shared_ptr<Tensor>(new Tensor(meta, this->_storage, this->_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &new_shape_in) const {
    // 元素个数必须一致（reshape不改变numel）
    const size_t old_numel = this->numel();
    size_t new_numel = 1;
    for (size_t d : new_shape_in) {
        if (d == 0) {
            throw std::invalid_argument("Tensor::view: new shape contains 0");
        }
        new_numel *= d;
    }
    if (old_numel != new_numel) {
        throw std::invalid_argument("Tensor::view: numel mismatch");
    }
    // 0维（标量）或只有一个元素：任意同numel的形状都可视为连续映射
    if (old_numel <= 1) {
        TensorMeta meta;
        meta.dtype = this->dtype();
        meta.shape = new_shape_in;
        meta.strides.assign(new_shape_in.size(), 0);

        ptrdiff_t st = 1;
        for (int i = static_cast<int>(new_shape_in.size()) - 1; i >= 0; --i) {
            meta.strides[i] = st;
            st *= static_cast<ptrdiff_t>(new_shape_in[i]);
        }

        return std::shared_ptr<Tensor>(new Tensor(meta, this->_storage, this->_offset));
    }
    // 从旧张量的shape/strides判断能否“无拷贝”reshape
    // 核心思想：忽略size==1的维度（它们不影响真实内存访问）
    // 把剩余维度按"stride递推关系是否成立"切成若干连续块chunk
    // 在同一个chunk中，内存是紧密连续的，可以任意拆分/合并（只要乘积相等）
    // new_shape也忽略size==1后，必须能按顺序恰好填好这些chunk的元素数
    // 这样可以正确处理题目中的反例： shape=(2,3,5), strides=(30,10,1)，因为 10 != 1*5(=5)，(3,5) 不是同一连续 chunk，不能合并成 15，所以应报错。
    const auto &old_shape = this->shape();
    const auto &old_strides = this->strides();

    // 压缩掉size==1维度
    struct Dim {
        size_t size;
        ptrdiff_t stride;
    };
    std::vector<Dim> dims;
    dims.reserve(old_shape.size());
    for (size_t i = 0; i < old_shape.size(); ++i) {
        if (old_shape[i] == 1) {
            continue;
        }
        dims.push_back(Dim{old_shape[i], old_strides[i]});
    }
    // 兜底：如果都被压缩没了（理论上numel>1不会发生），按contiguous处理
    if (dims.empty()) {
        TensorMeta meta; 
        meta.dtype = this->dtype();
        meta.shape = new_shape_in;
        meta.strides.assign(new_shape_in.size(), 0);

        ptrdiff_t st = 1;
        for (int i = static_cast<int>(new_shape_in.size()) - 1; i >= 0; --i) {
            meta.strides[i] = st;
            st *= static_cast<ptrdiff_t>(new_shape_in[i]);
        }
        return std::shared_ptr<Tensor>(new Tensor(meta, this->_storage, this->_offset));
    }
    // 把旧dims分成连续chunk（从外到内保存）
    std::vector<size_t> chunk_numels; // 每个chunk包含的元素数
    std::vector<ptrdiff_t> chunk_inner_stride; // 每个chunk最内层stride（用于生成新stride）

    {
        // 从最内层开始聚合
        size_t curr_numel = dims.back().size;
        ptrdiff_t curr_inner_stride = dims.back().stride;

        for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
            // 若 dims[i].stride == dims[i+1].stride * dims[i+1].size
            // 表示这两维在内存中是紧密连续递推关系，可以同属一个 chunk
            const ptrdiff_t expected = dims[i + 1].stride * static_cast<ptrdiff_t>(dims[i + 1].size);
            if (dims[i].stride == expected) {
                curr_numel *= dims[i].size;
            }
            else {
                // 断开：结束一个chunk
                chunk_numels.push_back(curr_numel);
                chunk_inner_stride.push_back(curr_inner_stride);
                // 开始新chunk
                curr_numel = dims[i].size;
                curr_inner_stride = dims[i].stride;
            }
        }
        // 最外层chunk
        chunk_numels.push_back(curr_numel);
        chunk_inner_stride.push_back(curr_inner_stride);
        // 目前是从内到外push的，翻转成从外到内
        std::reverse(chunk_numels.begin(), chunk_numels.end());
        std::reverse(chunk_inner_stride.begin(), chunk_inner_stride.end());
    }
    // new_shape同样忽略size==1
    std::vector<size_t> new_sizes;
    new_sizes.reserve(new_shape_in.size());
    for (size_t d : new_shape_in) {
        if (d == 1) {
            continue;
        }
        new_sizes.push_back(d);
    }
    // 逐个chunk匹配new_sizes:要求能按顺序把每个chunk的numel恰好乘出来
    std::vector<ptrdiff_t> new_strides_no1(new_sizes.size(), 0);

    size_t pos = 0; // new_sizes 的当前位置
    for (size_t c = 0; c < chunk_numels.size(); ++c) {
        const size_t target = chunk_numels[c];
        const ptrdiff_t inner = chunk_inner_stride[c];

        if (pos > new_sizes.size()) {
            throw std::runtime_error("Tensor::view: shape not compatible with strides");
        }
        size_t prod = 1;
        size_t k = pos;
        while (k < new_sizes.size() && prod < target) {
            prod *= new_sizes[k];
            ++k;
        }
        if (prod != target) {
            throw std::runtime_error("Tensor::view: shape not compatible with strides");
        }
        // 对 new_sizes[pos..k-1] 这一段生成 strides（chunk 内按 contiguous 推导，但以 inner 为最后一维 stride）
        ptrdiff_t st = inner;
        for (int i = static_cast<int>(k) - 1; i >= static_cast<int>(pos); --i) {
            new_strides_no1[i] = st;
            st *= static_cast<ptrdiff_t>(new_sizes[i]);
        }
        pos = k;
    }
    if (pos != new_sizes.size()) {
        // new_sizes 还剩维度没有对应到旧 chunk 的边界上
        throw std::runtime_error("Tensor::view: shape not compatible with strides");
    }
    // 把 size==1 的维度 stride 填回去：给一个合理值即可（不影响实际访问）
    // 我们用 contiguous 的默认 stride 来填 size==1 的维度。
    std::vector<ptrdiff_t> final_strides(new_shape_in.size(), 0);
    {
        std::vector<ptrdiff_t> contig(new_shape_in.size(), 0);
        ptrdiff_t st = 1;
        for (int i = static_cast<int>(new_shape_in.size()) - 1; i >= 0; --i) {
            contig[i] = st;
            st *= static_cast<ptrdiff_t>(new_shape_in[i]);
        }

        size_t p = 0;
        for (size_t i = 0; i < new_shape_in.size(); ++i) {
            if (new_shape_in[i] == 1) {
                final_strides[i] = contig[i];
            } else {
                final_strides[i] = new_strides_no1[p++];
            }
        }
    }
    // 构造新 Tensor（共享 storage + offset，不传输数据）
    TensorMeta meta;
    meta.dtype = this->dtype();
    meta.shape = new_shape_in;
    meta.strides = std::move(final_strides);
    return std::shared_ptr<Tensor>(new Tensor(meta, this->_storage, this->_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    const size_t n = this->ndim();
    if (dim >= n) {
        throw std::invalid_argument("Tensor::slice: dim out of range");
    }
    const auto &sh = this->shape();
    const auto &st = this->strides();
    // 边界检查：start/end必须形成合法区间[start, end)且落在该维范围内
    if (start > end) {
        throw std::invalid_argument("Tensor::slice: start must be <= end");
    }
    if (end > sh[dim]) {
        throw std::invalid_argument("Tensor::slice: end out of range");
    }
    // 新 meta：shape 只改 dim 这一维；strides 不变（仍然是同一块存储的同样步长）
    TensorMeta meta;
    meta.dtype = this->dtype();
    meta.shape = sh;
    meta.strides = st;
    meta.shape[dim] = end - start;
    // 关键：通过 offset 平移起点，实现“从 start 开始看”
    // 注意：这里假设 strides 的单位是“元素个数”（而不是字节数）
    const ptrdiff_t delta_elems = static_cast<ptrdiff_t>(start) * st[dim];
    if (delta_elems < 0) {
        throw std::runtime_error("Tensor::slice: negative offset delta");
    }
    const size_t new_offset = this->_offset
                            + static_cast<size_t>(delta_elems) * this->elementSize();
    return std::shared_ptr<Tensor>(new Tensor(meta, this->_storage, new_offset));
}

void Tensor::load(const void *src_) {
    if (src_ == nullptr) {
        throw std::invalid_argument("Tensor::load: src is nullptr");
    }
    core::context().setDevice(this->deviceType(), this->deviceId());
    const size_t nbytes = this->numel() * this->elementSize();
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(this->data(), src_, nbytes);
    } else {
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            nbytes,
            LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
