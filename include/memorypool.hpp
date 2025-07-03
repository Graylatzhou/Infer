#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <map>

class CudaMemoryPool {
public:
    CudaMemoryPool(const std::string& name = "default", size_t initial_size = 1ULL * 1024 * 1024 * 1024) 
        : name_(name), initialized_(false), device_id_(0), stream_(nullptr), total_size_(0) {
    }
    
    static CudaMemoryPool& getInstance() {
        static CudaMemoryPool instance;
        return instance;
    }

    bool initialize(cudaStream_t stream = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) return true;

        // 获取设备ID
        cudaError_t err = cudaGetDevice(&device_id_);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get device ID: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        // 打印CUDA版本信息
        int runtimeVer = 0, driverVer = 0;
        cudaRuntimeGetVersion(&runtimeVer);
        cudaDriverGetVersion(&driverVer);
        std::cout << "CUDA Runtime: " << runtimeVer/1000 << "." << (runtimeVer%100)/10 
                  << ", Driver: " << driverVer/1000 << "." << (driverVer%100)/10 << std::endl;
        
        size_t free, total;
        err = cudaMemGetInfo(&free, &total);
        if (err == cudaSuccess) {
            std::cout << "Memory Pool '" << name_ << "' initialized. GPU has " 
                    << (free/(1024*1024)) << " MB free out of " 
                    << (total/(1024*1024)) << " MB total" << std::endl;
        }

        stream_ = stream;
        initialized_ = true;
        return true;
    }

    // 分配内存，返回指针
    void* allocate(size_t size, cudaStream_t stream = nullptr) {
        if (size == 0) {
            std::cerr << "Cannot allocate zero bytes" << std::endl;
            return nullptr;
        }
        
        if (!initialized_) {
            if (!initialize(stream)) return nullptr;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        
        // 检查是否有缓存的内存块可用
        auto it = free_blocks_.lower_bound(size);
        if (it != free_blocks_.end()) {
            void* ptr = it->second;
            size_t block_size = it->first;
            free_blocks_.erase(it);
            allocated_blocks_[ptr] = block_size;
            
            if (block_size >= 10*1024*1024) { // 只记录较大的分配
                std::cout << "Memory Pool '" << name_ << "': Reusing " << (block_size/(1024*1024)) 
                        << " MB at " << ptr << std::endl;
            }
            return ptr;
        }

        // 没有合适的缓存块，直接分配新内存
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            size_t free, total;
            cudaMemGetInfo(&free, &total);
            
            std::cerr << "Failed to allocate memory: " << cudaGetErrorString(err) << std::endl;
            std::cerr << "GPU Memory: Free=" << (free/(1024*1024)) << "MB, Total=" << (total/(1024*1024)) 
                      << "MB, Requested=" << (size/(1024*1024)) << "MB" << std::endl;
            return nullptr;
        }
        
        allocated_blocks_[ptr] = size;
        total_size_ += size;
        
        if (size >= 10*1024*1024) { // 只记录较大的分配
            std::cout << "Memory Pool '" << name_ << "': Allocated " << (size/(1024*1024)) 
                    << " MB at " << ptr << ", total: " 
                    << (total_size_/(1024*1024)) << " MB" << std::endl;
        }
        
        return ptr;
    }

    void free(void* ptr) {
        if (!ptr || !initialized_) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = allocated_blocks_.find(ptr);
        if (it == allocated_blocks_.end()) {
            std::cerr << "Attempting to free unallocated memory at " << ptr << std::endl;
            return;
        }
        
        size_t size = it->second;
        
        // 策略：小块直接释放，大块缓存起来以便重用
        if (size >= 1024*1024) { // 1MB以上缓存
            free_blocks_.insert(std::make_pair(size, ptr));
            if (size >= 10*1024*1024) { // 只记录较大的释放
                std::cout << "Memory Pool '" << name_ << "': Cached " << (size/(1024*1024)) 
                        << " MB at " << ptr << std::endl;
            }
        } else {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                std::cerr << "Failed to free memory: " << cudaGetErrorString(err) << std::endl;
            }
            total_size_ -= size;
        }
        
        allocated_blocks_.erase(it);
    }

    cudaError_t copyAsync(void* dst, const void* src, size_t size, cudaMemcpyKind kind, cudaStream_t stream = nullptr) {
    // 基本参数验证
    if (!dst || !src) {
        std::cerr << "Error: Null pointer in copyAsync - dst: " << dst << ", src: " << src << std::endl;
        return cudaErrorInvalidValue;
    }
    
    if (size == 0) {
        return cudaSuccess; // 零大小复制视为成功
    }
    
    if (!initialized_ && !initialize(stream)) {
        return cudaErrorInitializationError;
    }
    
    // 在复制前检查CUDA状态
    cudaError_t pre_err = cudaGetLastError();
    if (pre_err != cudaSuccess) {
        std::cerr << "Warning: CUDA in error state before copyAsync: " << cudaGetErrorString(pre_err) << std::endl;
        // 可以选择是否继续或返回错误
    }
    
    // 确保之前的操作已完成
    if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyHostToDevice) {
        cudaStreamSynchronize(stream ? stream : stream_);
    }
    
    cudaStream_t use_stream = stream ? stream : stream_;
    cudaError_t err = cudaMemcpyAsync(dst, src, size, kind, use_stream);
    
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy memory asynchronously: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "Details - dst: " << dst << ", src: " << src
                  << ", size: " << size << " bytes, kind: " << kind << std::endl;
    }
    
    return err;
}
    
    // 打印内存池统计信息
    void printStats() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "=== Memory Pool Stats: " << name_ << " ===" << std::endl;
        std::cout << "Initialized: " << (initialized_ ? "Yes" : "No") << std::endl;
        if (initialized_) {
            std::cout << "Total Allocated: " << (total_size_ / (1024*1024)) << " MB" << std::endl;
            std::cout << "Active Allocations: " << allocated_blocks_.size() << std::endl;
            std::cout << "Cached Free Blocks: " << free_blocks_.size() << std::endl;
            
            size_t free, total;
            cudaError_t err = cudaMemGetInfo(&free, &total);
            if (err == cudaSuccess) {
                std::cout << "GPU Memory: Free=" << (free/(1024*1024)) << "MB, Total=" 
                          << (total/(1024*1024)) << "MB" << std::endl;
            }
        }
    }
    
    // 释放所有缓存的空闲块
    void releaseCache() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t released = 0;
        for (auto& pair : free_blocks_) {
            cudaError_t err = cudaFree(pair.second);
            if (err != cudaSuccess) {
                std::cerr << "Failed to free cached memory: " << cudaGetErrorString(err) << std::endl;
            }
            released += pair.first;
        }
        
        total_size_ -= released;
        free_blocks_.clear();
        
        std::cout << "Memory Pool '" << name_ << "': Released " << (released/(1024*1024)) 
                << " MB from cache" << std::endl;
    }

    ~CudaMemoryPool() {
        if (initialized_) {
            // 释放所有缓存的内存
            for (auto& pair : free_blocks_) {
                cudaFree(pair.second);
            }
            
            // 警告未释放的内存
            if (!allocated_blocks_.empty()) {
                std::cerr << "Warning: " << allocated_blocks_.size() << " allocations not freed!" << std::endl;
                for (auto& pair : allocated_blocks_) {
                    cudaFree(pair.first);
                }
            }
            
            std::cout << "Memory Pool '" << name_ << "' destroyed" << std::endl;
        }
    }

private:
    std::string name_;
    mutable std::mutex mutex_;
    bool initialized_;
    int device_id_;
    cudaStream_t stream_;
    size_t total_size_;
    
    std::unordered_map<void*, size_t> allocated_blocks_;    // 已分配的块：地址 -> 大小
    std::multimap<size_t, void*> free_blocks_;             // 空闲的块：大小 -> 地址（按大小排序）
};


class CudaMemoryManager {
public:
    static CudaMemoryManager& getInstance() {
        static CudaMemoryManager instance;
        return instance;
    }
    
    CudaMemoryPool& getWeightPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!weight_pool_) {
            weight_pool_ = new CudaMemoryPool("weight");
            weight_pool_->initialize();
        }
        return *weight_pool_;
    }
    
    CudaMemoryPool& getBufferPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!buffer_pool_) {
            buffer_pool_ = new CudaMemoryPool("buffer");
            buffer_pool_->initialize();
        }
        return *buffer_pool_;
    }
    
    // 打印所有内存池的统计信息
    void printAllStats() {
        if (weight_pool_) weight_pool_->printStats();
        if (buffer_pool_) buffer_pool_->printStats();
    }
    
    // 释放所有内存池的缓存
    void releaseAllCaches() {
        if (weight_pool_) weight_pool_->releaseCache();
        if (buffer_pool_) buffer_pool_->releaseCache();
    }
    
private:
    CudaMemoryManager() : weight_pool_(nullptr), buffer_pool_(nullptr) {}
    ~CudaMemoryManager() {
        delete weight_pool_;
        delete buffer_pool_;
    }
    
    // 禁止拷贝和赋值
    CudaMemoryManager(const CudaMemoryManager&) = delete;
    CudaMemoryManager& operator=(const CudaMemoryManager&) = delete;
    
    std::mutex mutex_;
    CudaMemoryPool* weight_pool_;
    CudaMemoryPool* buffer_pool_;
};
