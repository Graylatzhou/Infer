#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <memory>
#include <string>

class CudaStreamMemoryPool {
public:

    CudaStreamMemoryPool(const std::string& name = "default") : name_(name), initialized_(false), device_id_(0), stream_(nullptr) {}
    
    static CudaStreamMemoryPool& getInstance() {
        static CudaStreamMemoryPool instance;
        return instance;
    }

    cudaMemPool_t getMemPoolHandle() const {
        return mem_pool_;
    }

    bool initialize(cudaStream_t stream = nullptr, size_t max_size = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) return true;

        // 获取当前设备
        cudaError_t err = cudaGetDevice(&device_id_);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get device ID: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        // 创建内存池
        cudaMemPoolProps poolProps = {};
        poolProps.allocType = cudaMemAllocationTypePinned;
        poolProps.location.type = cudaMemLocationTypeDevice;
        poolProps.location.id = device_id_;
        poolProps.handleTypes = cudaMemHandleTypeNone;

        err = cudaMemPoolCreate(&mem_pool_, &poolProps);
        if (err != cudaSuccess) {
            std::cerr << "Failed to create memory pool: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        // 设置内存池属性（可选）
        if (max_size > 0) {
            // 设置最大内存大小
            err = cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolAttrReleaseThreshold, &max_size);
            if (err != cudaSuccess) {
                std::cerr << "Failed to set pool max size: " << cudaGetErrorString(err) << std::endl;
            }
        }

        // 存储流以供后续使用
        stream_ = stream;
        initialized_ = true;
        return true;
    }

    void* allocate(size_t size, cudaStream_t stream = nullptr) {
        if (!initialized_) {
            if (!initialize(stream)) return nullptr;
        }

        cudaStream_t use_stream = stream ? stream : stream_;
        void* ptr = nullptr;
        
        // 从池中分配内存
        cudaError_t err = cudaMallocFromPoolAsync(&ptr, size, mem_pool_, use_stream);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate from pool: " << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        
        return ptr;
    }

    void free(void* ptr, cudaStream_t stream = nullptr) {
        if (!ptr) return;
        
        cudaStream_t use_stream = stream ? stream : stream_;
        cudaError_t err = cudaFreeAsync(ptr, use_stream);
        if (err != cudaSuccess) {
            std::cerr << "Failed to free memory: " << cudaGetErrorString(err) << std::endl;
        }
    }

    cudaError_t copyAsync(void* dst, const void* src, size_t size, cudaMemcpyKind kind, cudaStream_t stream = nullptr) {
        if (!initialized_ && !initialize(stream)) {
            return cudaErrorInitializationError;
        }
        
        cudaStream_t use_stream = stream ? stream : stream_;
        cudaError_t err = cudaMemcpyAsync(dst, src, size, kind, use_stream);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy memory asynchronously: " << cudaGetErrorString(err) << std::endl;
        }
        return err;
    }

    ~CudaStreamMemoryPool() {
        if (initialized_) {
            cudaMemPoolDestroy(mem_pool_);
        }
    }

private:

    std::string name_;
    mutable std::mutex mutex_;
    bool initialized_;
    int device_id_;
    cudaMemPool_t mem_pool_;
    cudaStream_t stream_;
};


class CudaMemoryPoolManager {
    public:
        static CudaMemoryPoolManager& getInstance() {
            static CudaMemoryPoolManager instance;
            return instance;
        }
        
        // 访问不同用途的内存池
        CudaStreamMemoryPool& getWeightPool() {
            if (!persistent_pool_) {
                persistent_pool_ = new CudaStreamMemoryPool("persistent");
                persistent_pool_->initialize();
                
                // 为持久池设置较大的释放阈值，减少内存被归还系统的可能性
                uint64_t threshold = 1ULL * 1024 * 1024 * 1024;
                cudaMemPoolSetAttribute(persistent_pool_->getMemPoolHandle(),
                                        cudaMemPoolAttrReleaseThreshold, &threshold);
            }
            return *persistent_pool_;
        }
        
        CudaStreamMemoryPool& getTemporaryPool() {
            if (!temporary_pool_) {
                temporary_pool_ = new CudaStreamMemoryPool("temporary");
                temporary_pool_->initialize();
            }
            return *temporary_pool_;
        }
        
    private:
        CudaMemoryPoolManager() : persistent_pool_(nullptr), temporary_pool_(nullptr) {}
        ~CudaMemoryPoolManager() {
            delete persistent_pool_;
            delete temporary_pool_;
        }
        
        CudaStreamMemoryPool* persistent_pool_;
        CudaStreamMemoryPool* temporary_pool_;
    };
