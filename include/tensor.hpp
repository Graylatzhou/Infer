#pragma once
#include <vector>
#include <memorypool.hpp>
#include <cuda_bf16.h>

namespace infer {
enum class Device {
    CPU,
    CUDA
};

enum class Usage {
    Weight,
    Buffer
};

struct SliceParam {
    size_t start;
    size_t end;
    size_t dim;
};


template <typename T>
class Tensor {
public:
    using Shape = std::vector<size_t>;
    using Stride = std::vector<size_t>;
    //构造空张量
    Tensor() 
     : shape_({}), device_(Device::CPU), stride_({}), data_(nullptr), gpu_data_(nullptr), total_size_(0), offset_(0), tag_(Usage::Buffer){}
    // 构造指定形状的张量
    // tag分为weight和temporary两种
    Tensor(const std::vector<size_t>& shape, Device device, cudaStream_t stream = nullptr)
     : shape_(shape), device_(device), stream_(stream), tag_(Usage::Buffer) {
        stride_.resize(shape.size());
        size_t total_size = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            stride_[i] = total_size;
            total_size *= shape[i];
        }
        total_size_ = total_size;
        if (device_ == Device::CPU) {
            data_ = std::make_shared<std::vector<T>>(total_size);
            gpu_data_ = nullptr;
        } else if (device_ == Device::CUDA) {
            data_ = nullptr;
            T* gpu_ptr = nullptr;
            gpu_ptr = static_cast<T*>(CudaMemoryManager::getInstance().getBufferPool().allocate(total_size * sizeof(T), stream_));
            gpu_data_ = std::shared_ptr<T>(gpu_ptr, [](T* ptr) {
                CudaMemoryManager::getInstance().getBufferPool().free(ptr);
            });
        } else {
            throw std::runtime_error("Unsupported device type"); 
        }
    }

    Tensor (const std::vector<size_t>& shape, Device device, void* weight_data, cudaStream_t stream = nullptr)
    : shape_(shape), device_(device), stream_(stream), tag_(Usage::Weight) {
        stride_.resize(shape.size());
        size_t total_size = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            stride_[i] = total_size;
            total_size *= shape[i];
        }
        total_size_ = total_size;
        if (device_ == Device::CPU) {
            data_ = std::make_shared<std::vector<T>>(total_size);
            gpu_data_ = nullptr;
            memcpy(data_->data(), weight_data, total_size * sizeof(T));
        } else if (device_ == Device::CUDA) {
            data_ = nullptr;
            T* gpu_ptr = static_cast<T*>(CudaMemoryManager::getInstance().getWeightPool().allocate(total_size * sizeof(T), stream_));
            gpu_data_ = std::shared_ptr<T>(gpu_ptr, [](T* ptr) {
                CudaMemoryManager::getInstance().getWeightPool().free(ptr);
            });
            CudaMemoryManager::getInstance().getWeightPool().copyAsync(gpu_data_.get(), weight_data, total_size * sizeof(T), cudaMemcpyHostToDevice, stream_);
        } else {
            throw std::runtime_error("Unsupported device type");
        }
    }

    ~Tensor() {

    }

    T* data_ptr() {
        if (device_ == Device::CPU) {
            return data_->data() + offset_;
        } else if (device_ == Device::CUDA) {
            return gpu_data_.get() + offset_;
        }
        return nullptr;
    }

    const T* data_ptr() const {
        if (device_ == Device::CPU) {
            return data_->data() + offset_;
        } else {
            return gpu_data_.get() + offset_;
        }
    }

    void fill(T value) {
        if (device_ == Device::CPU) {
            std::fill(data_->begin(), data_->end(), value);
        } else if (device_ == Device::CUDA) {
            std::vector<T> host_data(total_size_, value);
            if (tag_ == Usage::Weight) {
                CudaMemoryManager::getInstance().getWeightPool().copyAsync(gpu_data_.get(), host_data.data(), total_size_ * sizeof(T), cudaMemcpyHostToDevice, stream_);
            } else {
                CudaMemoryManager::getInstance().getBufferPool().copyAsync(gpu_data_.get(), host_data.data(), total_size_ * sizeof(T), cudaMemcpyHostToDevice, stream_);
            }
        }
    }

    const Shape& shape() const {
        return shape_;
    }

    const Stride& stride() const {
        return stride_;
    }


    void to(Device new_device) const {
        if (device_ == Device::CPU && new_device == Device::CUDA) {
            // CPU -> CUDA
            if (tag_ == Usage::Weight) {
                CudaMemoryManager::getInstance().getWeightPool().copyAsync(
                    gpu_data_.get(), data_->data() + offset_, 
                    total_size_ * sizeof(T), cudaMemcpyHostToDevice, stream_);
            } else {
                CudaMemoryManager::getInstance().getBufferPool().copyAsync(
                    gpu_data_.get(), data_->data() + offset_, 
                    total_size_ * sizeof(T), cudaMemcpyHostToDevice, stream_);
            }
        } else if (device_ == Device::CUDA && new_device == Device::CPU) {
            // CUDA -> CPU
            if (tag_ == Usage::Weight) {
                CudaMemoryManager::getInstance().getWeightPool().copyAsync(
                    data_->data(), gpu_data_.get() + offset_, 
                    total_size_ * sizeof(T), cudaMemcpyDeviceToHost, stream_);
            } else {
                CudaMemoryManager::getInstance().getBufferPool().copyAsync(
                    data_->data(), gpu_data_.get() + offset_, 
                    total_size_ * sizeof(T), cudaMemcpyDeviceToHost, stream_);
            }
        }
        return;
    }

    static Tensor<T> Buffer(const std::vector<size_t> shape, Device device, cudaStream_t stream = nullptr) {
        Tensor<T> tensor(shape, device, stream);
        tensor.tag_ = Usage::Buffer;
        return tensor;
    }

    static Tensor<T> Weight(const std::vector<size_t> shape, Device device, void* data, cudaStream_t stream = nullptr) {
        Tensor<T> tensor(shape, device, data, stream);
        tensor.tag_ = Usage::Weight;
        return tensor;
    }


    Tensor<T> clone() const {
        // 创建一个新的空张量，具有相同形状和设备类型
        Tensor<T> new_tensor(shape_, device_, stream_);
        new_tensor.tag_ = tag_; // 设置tag
        
        // 复制数据
        if (device_ == Device::CPU) {
            std::copy(data_->begin(), data_->end(), new_tensor.data_->begin());
        } else if (device_ == Device::CUDA) {
            // 直接使用新张量已分配的内存
            if (tag_ == Usage::Weight) {
                CudaMemoryManager::getInstance().getWeightPool().copyAsync(
                    new_tensor.gpu_data_.get(), gpu_data_.get(), 
                    total_size_ * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            } else {
                CudaMemoryManager::getInstance().getBufferPool().copyAsync(
                    new_tensor.gpu_data_.get(), gpu_data_.get(), 
                    total_size_ * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            }
        }
        
        return new_tensor;
    }

    Device device() const {
        return device_;
    }

    size_t size() const {
        return total_size_;
    }

    size_t offset() const {
        return offset_;
    }

    Usage getTag() const {
        return tag_;
    }

    size_t ndim() const {
        return shape_.size();
    }

    const cudaStream_t& getStream() const {
        return stream_;
    }

    Tensor operator[](const std::vector<SliceParam> &slice_params) const { // slice
        for (const auto& param : slice_params) {
            if (param.start >= param.end || param.end > shape_[param.dim]) {
                throw std::out_of_range("Slice indices are out of range");
            }
        }
        Tensor slice_tensor(*this);
        for (const auto& param : slice_params) {
            slice_tensor.shape_[param.dim] = param.end - param.start;
            slice_tensor.offset_ += calcOffset(param.dim, param.start);
        }
        return slice_tensor;
    }

    Tensor view(const Shape &new_shape) & {
        size_t new_total_size = 1;
        for (auto dim_size : new_shape) {
            new_total_size *= dim_size;
        }
        if (new_total_size != total_size_) {
            throw std::invalid_argument("Incompatible shape");
        }
        this->shape_ = new_shape;
        this->stride_ = calcStride(new_shape);
        return *this;
    }

    Tensor permute(const std::vector<size_t>& new_order) const {
        if (new_order.size() != shape_.size()) {
            throw std::invalid_argument("New order size must match tensor dimension");
        }
        Tensor permuted_tensor(*this);
        Shape new_shape(shape_.size());
        Stride new_stride(shape_.size());
        for (size_t i = 0; i < new_order.size(); ++i) {
            new_shape[i] = shape_[new_order[i]];
            new_stride[i] = stride_[new_order[i]];
        }
        permuted_tensor.shape_ = new_shape;
        permuted_tensor.stride_ = new_stride;
        return permuted_tensor;
    }

private:
    Shape shape_;
    Stride stride_;
    Device device_;
    std::shared_ptr<std::vector<T>> data_;
    std::shared_ptr<T> gpu_data_;
    Usage tag_;
    size_t offset_;
    size_t total_size_;
    cudaStream_t stream_;

    static inline void checkCudaError(cudaError_t err) {
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
        }
    }

    size_t calcOffset(const size_t dim, const size_t start) const {
        if (dim >= shape_.size()) {
            throw std::out_of_range("Dimension out of range");
        }
        return stride_[dim] * start;
    }

    Stride calcStride(const Shape& shape) const {
        Stride stride(shape.size());
        size_t total_size = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            stride[i] = total_size;
            total_size *= shape[i];
        }
        return stride;
    }

};
};