#pragma once
#include <vector>
#include <memorypool.hpp>

namespace infer {
enum class Device {
    CPU,
    CUDA
};

template <typename T>
class Tensor {
public:
    using Shape = std::vector<size_t>;
    using Stride = std::vector<size_t>;
    //构造空张量
    Tensor() 
     : shape_({}), device_(Device::CPU), stride_({}), data_(nullptr), gpu_data_(nullptr), tag(""), total_size_(0), offset_(0) {}
    // 构造指定形状的张量
    // tag分为weight和temporary两种
    Tensor(const std::vector<size_t>& shape, Device device, const std::string& tag = "", cudaStream_t stream = nullptr)
     : shape_(shape), device_(device), tag(tag), offset_(0), stream_(stream) {
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
            if (tag == "weight") {
                gpu_ptr = static_cast<T*>(CudaMemoryPoolManager::getInstance().getWeightPool().allocate(total_size * sizeof(T), stream_));
            } else {
                gpu_ptr = static_cast<T*>(CudaMemoryPoolManager::getInstance().getTemporaryPool().allocate(total_size * sizeof(T), stream_));
            }
            gpu_data_ = std::shared_ptr<T>(gpu_ptr, [this](T* ptr) {
                if (this->tag == "weight") {
                    CudaMemoryPoolManager::getInstance().getWeightPool().free(ptr, stream_);
                } else {
                    CudaMemoryPoolManager::getInstance().getTemporaryPool().free(ptr, stream_);
                }
            });
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
            return data_->data();
        } else {
            return gpu_data_.get();
        }
    }

    void fill(T value) {
        if (device_ == Device::CPU) {
            std::fill(data_->begin(), data_->end(), value);
        } else if (device_ == Device::CUDA) {
            std::vector<T> host_data(total_size_, value);
            if (tag == "weight") {
                CudaMemoryPoolManager::getInstance().getWeightPool().copyAsync(gpu_data_.get(), host_data.data(), total_size_ * sizeof(T), cudaMemcpyHostToDevice, stream_);
            } else {
                CudaMemoryPoolManager::getInstance().getTemporaryPool().copyAsync(gpu_data_.get(), host_data.data(), total_size_ * sizeof(T), cudaMemcpyHostToDevice, stream_);
            }
        }
    }

    const Shape& shape() const {
        return shape_;
    }

    const Stride& stride() const {
        return stride_;
    }


    Tensor<T>& to(Device newDevice) {
        if (device_ == newDevice) return *this;
        if (newDevice == Device::CUDA && device_ == Device::CPU) {
            if (gpu_data_) {
                if (tag == "weight") {
                    CudaMemoryPoolManager::getInstance().getWeightPool().copyAsync(
                        gpu_data_.get(), data_->data(), 
                        total_size_ * sizeof(T), cudaMemcpyHostToDevice, stream_); // 添加stream_
                } else {
                    CudaMemoryPoolManager::getInstance().getTemporaryPool().copyAsync(
                        gpu_data_.get(), data_->data(), 
                        total_size_ * sizeof(T), cudaMemcpyHostToDevice, stream_); // 添加stream_
                }
            }
        } else if (newDevice == Device::CPU && device_ == Device::CUDA) {
            if (data_) {
                if (tag == "weight") {
                    CudaMemoryPoolManager::getInstance().getWeightPool().copyAsync(data_->data(), gpu_data_.get(), total_size_ * sizeof(T), cudaMemcpyDeviceToHost, this->stream_);
                } else {
                    CudaMemoryPoolManager::getInstance().getTemporaryPool().copyAsync(data_->data(), gpu_data_.get(), total_size_ * sizeof(T), cudaMemcpyDeviceToHost, this->stream_);
                }
            }
        }
        device_ = newDevice;
        return *this;
    }

    Tensor<T> clone() const {
        // 创建一个新的空张量，具有相同形状和设备类型
        Tensor<T> new_tensor(shape_, device_, tag, stream_);
        
        // 复制数据
        if (device_ == Device::CPU) {
            std::copy(data_->begin(), data_->end(), new_tensor.data_->begin());
        } else if (device_ == Device::CUDA) {
            // 直接使用新张量已分配的内存
            if (tag == "weight") {
                CudaMemoryPoolManager::getInstance().getWeightPool().copyAsync(
                    new_tensor.gpu_data_.get(), gpu_data_.get(), 
                    total_size_ * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
            } else {
                CudaMemoryPoolManager::getInstance().getTemporaryPool().copyAsync(
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

    const std::string& getTag() const {
        return tag;
    }

    size_t dim() const {
        return shape_.size();
    }

    const cudaStream_t& getStream() const {
        return stream_;
    }

private:
    Shape shape_;
    Stride stride_;
    Device device_;
    std::shared_ptr<std::vector<T>> data_;
    std::shared_ptr<T> gpu_data_;
    std::string tag; //用于标记是否为权重
    size_t offset_;
    size_t total_size_;
    cudaStream_t stream_;

    static inline void checkCudaError(cudaError_t err) {
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
        }
    }

};
};