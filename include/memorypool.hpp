#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cuda.h>

class CudaVirtualMemoryPool {
public:
    CudaVirtualMemoryPool(const std::string& name = "default", size_t initial_size = 1ULL * 1024 * 1024 * 1024) 
        : name_(name), initialized_(false), device_id_(0), stream_(nullptr), 
          virtual_address_(nullptr), total_reserved_size_(initial_size), used_size_(0) {
    }
    
    static CudaVirtualMemoryPool& getInstance() {
        static CudaVirtualMemoryPool instance;
        return instance;
    }

    bool initialize(cudaStream_t stream = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) return true;

        // 初始化CUDA驱动API
        CUresult res = cuInit(0);
        if (res != CUDA_SUCCESS) {
            const char *errorName, *errorString;
            cuGetErrorName(res, &errorName);
            cuGetErrorString(res, &errorString);
            std::cerr << "Failed to initialize CUDA driver: " << errorName << " - " << errorString << std::endl;
            return false;
        }

        cudaError_t err = cudaGetDevice(&device_id_);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get device ID: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        // 保留虚拟内存地址空间
        res = cuMemAddressReserve((CUdeviceptr*)&virtual_address_, total_reserved_size_, 0, 0, 0);
        if (res != CUDA_SUCCESS) {
            const char *errorName, *errorString;
            cuGetErrorName(res, &errorName);
            cuGetErrorString(res, &errorString);
            std::cerr << "Failed to reserve virtual memory: " << errorName << " - " << errorString << std::endl;
            return false;
        }

        stream_ = stream;
        initialized_ = true;
        return true;
    }

    // 分配内存，返回虚拟地址
    void* allocate(size_t size, cudaStream_t stream = nullptr) {
        if (!initialized_) {
            if (!initialize(stream)) return nullptr;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        
        // 对齐到页大小
        constexpr size_t page_size = 64 * 1024; // 64 KB CUDA 页大小
        size = (size + page_size - 1) & ~(page_size - 1);

        // 检查可用空间
        if (used_size_ + size > total_reserved_size_) {
            // 尝试寻找空闲块
            auto fit_block = findFreeBlock(size);
            if (fit_block != free_blocks_.end()) {
                void* block_addr = fit_block->first;
                size_t block_size = fit_block->second;
                
                // 如果找到刚好大小的块，移除它
                if (block_size == size) {
                    free_blocks_.erase(fit_block);
                } else {
                    // 否则缩小块大小并调整起始地址
                    free_blocks_.erase(fit_block);
                    free_blocks_[static_cast<char*>(block_addr) + size] = block_size - size;
                }
                
                // 记录分配的块
                allocated_blocks_[block_addr] = size;
                return block_addr;
            }
            
            std::cerr << "Out of memory in VMM pool" << std::endl;
            return nullptr;
        }

        // 分配物理内存
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id_;
        
        CUmemGenericAllocationHandle mem_handle;
        CUresult res = cuMemCreate(&mem_handle, size, &prop, 0);
        if (res != CUDA_SUCCESS) {
            const char *errorName, *errorString;
            cuGetErrorName(res, &errorName);
            cuGetErrorString(res, &errorString);
            std::cerr << "Failed to allocate physical memory: " << errorName << " - " << errorString << std::endl;
            return nullptr;
        }
        
        // 映射物理内存到虚拟地址空间
        void* block_addr = static_cast<char*>(virtual_address_) + used_size_;
        res = cuMemMap((CUdeviceptr)block_addr, size, 0, mem_handle, 0);
        if (res != CUDA_SUCCESS) {
            const char *errorName, *errorString;
            cuGetErrorName(res, &errorName);
            cuGetErrorString(res, &errorString);
            std::cerr << "Failed to map memory: " << errorName << " - " << errorString << std::endl;
            cuMemRelease(mem_handle);
            return nullptr;
        }
        
        // 设置内存访问权限
        CUmemAccessDesc access_desc = {};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device_id_;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        
        res = cuMemSetAccess((CUdeviceptr)block_addr, size, &access_desc, 1);
        if (res != CUDA_SUCCESS) {
            const char *errorName, *errorString;
            cuGetErrorName(res, &errorName);
            cuGetErrorString(res, &errorString);
            std::cerr << "Failed to set memory access: " << errorName << " - " << errorString << std::endl;
            cuMemUnmap((CUdeviceptr)block_addr, size);
            cuMemRelease(mem_handle);
            return nullptr;
        }
        
        // 记录此次分配
        allocated_blocks_[block_addr] = size;
        memory_handles_[block_addr] = mem_handle;
        
        // 更新已用大小
        used_size_ += size;
        
        return block_addr;
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
        
        // 取消映射
        CUresult res = cuMemUnmap((CUdeviceptr)ptr, size);
        if (res != CUDA_SUCCESS) {
            const char *errorName, *errorString;
            cuGetErrorName(res, &errorName);
            cuGetErrorString(res, &errorString);
            std::cerr << "Failed to unmap memory: " << errorName << " - " << errorString << std::endl;
        }
        
        // 释放物理内存
        auto handle_it = memory_handles_.find(ptr);
        if (handle_it != memory_handles_.end()) {
            res = cuMemRelease(handle_it->second);
            if (res != CUDA_SUCCESS) {
                const char *errorName, *errorString;
                cuGetErrorName(res, &errorName);
                cuGetErrorString(res, &errorString);
                std::cerr << "Failed to release memory: " << errorName << " - " << errorString << std::endl;
            }
            memory_handles_.erase(handle_it);
        }
        
        // 添加到空闲块列表，允许后续复用
        mergeAndAddFreeBlock(ptr, size);
        
        // 从已分配列表移除
        allocated_blocks_.erase(it);
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

    ~CudaVirtualMemoryPool() {
        if (initialized_) {
            // 释放所有分配的内存
            for (auto& pair : memory_handles_) {
                cuMemRelease(pair.second);
            }
            
            // 释放虚拟地址空间
            cuMemAddressFree((CUdeviceptr)virtual_address_, total_reserved_size_);
        }
    }

private:
    // 寻找合适大小的空闲块
    std::unordered_map<void*, size_t>::iterator findFreeBlock(size_t size) {
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            if (it->second >= size) {
                return it;
            }
        }
        return free_blocks_.end();
    }
    
    // 合并相邻空闲块并添加新的空闲块
    void mergeAndAddFreeBlock(void* ptr, size_t size) {
        // 检查前一个块
        void* prev_end = nullptr;
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            char* block_end = static_cast<char*>(it->first) + it->second;
            if (block_end == ptr) {
                // 可以向前合并
                it->second += size;
                
                // 检查是否可以向后合并
                char* new_end = static_cast<char*>(it->first) + it->second;
                auto next_it = free_blocks_.find(new_end);
                if (next_it != free_blocks_.end()) {
                    it->second += next_it->second;
                    free_blocks_.erase(next_it);
                }
                
                return;
            }
        }
        
        // 检查后一个块
        char* this_end = static_cast<char*>(ptr) + size;
        auto next_it = free_blocks_.find(this_end);
        if (next_it != free_blocks_.end()) {
            // 向后合并
            size_t merged_size = size + next_it->second;
            free_blocks_.erase(next_it);
            free_blocks_[ptr] = merged_size;
            return;
        }
        
        // 没有可以合并的，添加新块
        free_blocks_[ptr] = size;
    }
    
    std::string name_;
    mutable std::mutex mutex_;
    bool initialized_;
    int device_id_;
    cudaStream_t stream_;
    
    void* virtual_address_;             // 虚拟内存起始地址
    size_t total_reserved_size_;        // 保留的总虚拟空间大小
    size_t used_size_;                  // 已使用的虚拟空间大小
    
    std::unordered_map<void*, size_t> allocated_blocks_;    // 已分配的块：地址 -> 大小
    std::unordered_map<void*, size_t> free_blocks_;         // 空闲的块：地址 -> 大小
    std::unordered_map<void*, CUmemGenericAllocationHandle> memory_handles_; // 内存句柄映射
};


class CudaVirtualMemoryManager {
public:
    static CudaVirtualMemoryManager& getInstance() {
        static CudaVirtualMemoryManager instance;
        return instance;
    }
    
    CudaVirtualMemoryPool& getWeightPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!weight_pool_) {
            weight_pool_ = new CudaVirtualMemoryPool("weight", 4ULL * 1024 * 1024 * 1024); // 4 GB
            weight_pool_->initialize();
        }
        return *weight_pool_;
    }
    
    CudaVirtualMemoryPool& getBufferPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!buffer_pool_) {
            buffer_pool_ = new CudaVirtualMemoryPool("buffer", 2ULL * 1024 * 1024 * 1024); // 2 GB
            buffer_pool_->initialize();
        }
        return *buffer_pool_;
    }
    
private:
    CudaVirtualMemoryManager() : weight_pool_(nullptr), buffer_pool_(nullptr) {}
    ~CudaVirtualMemoryManager() {
        delete weight_pool_;
        delete buffer_pool_;
    }
    
    // 禁止拷贝和赋值
    CudaVirtualMemoryManager(const CudaVirtualMemoryManager&) = delete;
    CudaVirtualMemoryManager& operator=(const CudaVirtualMemoryManager&) = delete;
    
    std::mutex mutex_;
    CudaVirtualMemoryPool* weight_pool_;
    CudaVirtualMemoryPool* buffer_pool_;
};
