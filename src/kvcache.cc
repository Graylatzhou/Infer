#include "kvcache.hpp"
#include <iostream>
#include <cassert>

namespace infer {

template<typename T>
KVCachePool<T>::KVCachePool(const KVCachePoolConfig& config, cudaStream_t stream)
    : config_(config), stream_(stream) {
    
    // 初始化统计信息
    stats_.total_allocated = 0;
    stats_.active_sequences = 0;
    stats_.max_sequence_len = 0;
    
    std::cout << "KV Cache Pool initialized with config:" << std::endl;
    std::cout << "  Block size: " << config.block_size << " tokens" << std::endl;
    std::cout << "  Num layers: " << config.num_layers << std::endl;
    std::cout << "  Num heads: " << config.num_heads << std::endl;
    std::cout << "  Head size: " << config.head_size << std::endl;
    std::cout << "  MLA: " << (config.use_mla ? "enabled" : "disabled") << std::endl;
}

template<typename T>
KVCachePool<T>::~KVCachePool() {
    // 释放所有序列
    for (auto& pair : sequence_caches_) {
        free_sequence(pair.first);
    }
}

template<typename T>
Tensor<T> KVCachePool<T>::get_key_cache(size_t sequence_id, size_t layer_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sequence_caches_.find(sequence_id);
    if (it == sequence_caches_.end()) {
        throw std::runtime_error("Sequence ID not found: " + std::to_string(sequence_id));
    }
    
    if (layer_idx >= it->second.key_caches.size() || it->second.key_caches[layer_idx] == nullptr) {
        throw std::runtime_error("Layer index out of bounds or cache not initialized: " + std::to_string(layer_idx));
    }
    
    return *(it->second.key_caches[layer_idx]);
}

template<typename T>
Tensor<T> KVCachePool<T>::get_value_cache(size_t sequence_id, size_t layer_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sequence_caches_.find(sequence_id);
    if (it == sequence_caches_.end()) {
        throw std::runtime_error("Sequence ID not found: " + std::to_string(sequence_id));
    }
    
    // 如果使用MLA，返回key缓存
    if (config_.use_mla) {
        if (layer_idx >= it->second.key_caches.size() || it->second.key_caches[layer_idx] == nullptr) {
            throw std::runtime_error("Layer index out of bounds or cache not initialized: " + std::to_string(layer_idx));
        }
        return *(it->second.key_caches[layer_idx]);
    }
    
    if (layer_idx >= it->second.value_caches.size() || it->second.value_caches[layer_idx] == nullptr) {
        throw std::runtime_error("Layer index out of bounds or cache not initialized: " + std::to_string(layer_idx));
    }
    
    return *(it->second.value_caches[layer_idx]);
}

template<typename T>
void KVCachePool<T>::extend_sequence(size_t sequence_id, size_t new_length) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sequence_caches_.find(sequence_id);
    if (it == sequence_caches_.end()) {
        // 如果序列不存在，则创建
        prefetch_sequence(sequence_id, new_length);
        return;
    }
    
    auto& cache_info = it->second;
    if (new_length <= cache_info.allocated_length) {
        // 如果已分配空间足够，只更新长度
        cache_info.current_length = new_length;
        return;
    }
    
    // 调整张量大小
    resize_cache_tensors(cache_info, new_length);
    
    // 更新分配长度和当前长度
    cache_info.allocated_length = new_length;
    cache_info.current_length = new_length;
    
    // 更新统计信息
    stats_.max_sequence_len = std::max(stats_.max_sequence_len, new_length);
}

template<typename T>
void KVCachePool<T>::free_sequence(size_t sequence_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sequence_caches_.find(sequence_id);
    if (it == sequence_caches_.end()) {
        return; // 序列不存在，无需操作
    }
    
    // 计算释放的内存大小
    size_t freed_memory = 0;
    auto& cache_info = it->second;
    
    // 每个key和value缓存的大小
    size_t tensor_size = cache_info.allocated_length * config_.num_heads * 
                        config_.head_size * sizeof(T);
    
    // 释放Key缓存
    for (auto* key_cache : cache_info.key_caches) {
        if (key_cache != nullptr) {
            freed_memory += tensor_size;
            delete key_cache;  // 释放内存
        }
    }
    
    for (auto* value_cache : cache_info.value_caches) {
        if (value_cache != nullptr) {
            freed_memory += tensor_size;
            delete value_cache;  // 释放内存
        }
    }

    
    // 更新统计信息
    stats_.total_allocated -= freed_memory;
    stats_.active_sequences--;
    
    // 移除序列
    sequence_caches_.erase(it);
}

template<typename T>
void KVCachePool<T>::prefetch_sequence(size_t sequence_id, size_t initial_length) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 检查序列是否已存在
    if (sequence_caches_.find(sequence_id) != sequence_caches_.end()) {
        // 已存在，调用扩展方法
        extend_sequence(sequence_id, initial_length);
        return;
    }
    
    // 创建新序列缓存信息
    SequenceCacheInfo cache_info;
    cache_info.current_length = initial_length;
    cache_info.allocated_length = initial_length;
    
    // 为每个层分配key缓存指针空间
    cache_info.key_caches.resize(config_.num_layers, nullptr);
    cache_info.value_caches.resize(config_.num_layers, nullptr);
    
    // 计算分配的内存
    size_t allocated_memory = 0;
    
    // 创建shape
    std::vector<size_t> shape = {
        initial_length,
        config_.num_heads,
        config_.head_size
    };
    
    // 为每层分配key缓存
    for (size_t layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        // 创建新张量并分配内存
        cache_info.key_caches[layer_idx] = new Tensor<T>(shape, Device::CUDA, stream_);
        // 初始化为0
        cache_info.key_caches[layer_idx]->fill(static_cast<T>(0));
        allocated_memory += initial_length * config_.num_heads * config_.head_size * sizeof(T);
        
        cache_info.value_caches[layer_idx] = new Tensor<T>(shape, Device::CUDA, stream_);
        // 初始化为0
        cache_info.value_caches[layer_idx]->fill(static_cast<T>(0));
        allocated_memory += initial_length * config_.num_heads * config_.head_size * sizeof(T);
    }
    
    // 添加序列到映射表
    sequence_caches_[sequence_id] = std::move(cache_info);
    
    // 更新统计信息
    stats_.total_allocated += allocated_memory;
    stats_.active_sequences++;
    stats_.max_sequence_len = std::max(stats_.max_sequence_len, initial_length);
}

template<typename T>
void KVCachePool<T>::print_memory_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "KV Cache Pool Memory Stats:" << std::endl;
    std::cout << "  Total Allocated Memory: " << (stats_.total_allocated / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Active Sequences: " << stats_.active_sequences << std::endl;
    std::cout << "  Max Sequence Length: " << stats_.max_sequence_len << std::endl;
    
    // 计算每个序列的平均长度
    size_t total_length = 0;
    for (const auto& pair : sequence_caches_) {
        total_length += pair.second.current_length;
    }
    
    if (stats_.active_sequences > 0) {
        double avg_length = static_cast<double>(total_length) / stats_.active_sequences;
        std::cout << "  Average Sequence Length: " << avg_length << " tokens" << std::endl;
    }
}

template<typename T>
void KVCachePool<T>::optimize_memory() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 简单实现：尝试缩减超额分配的序列
    for (auto& pair : sequence_caches_) {
        auto& cache_info = pair.second;
        
        // 如果分配长度远大于当前长度，考虑缩减
        if (cache_info.allocated_length > cache_info.current_length * 1.5 && 
            cache_info.allocated_length - cache_info.current_length > config_.block_size) {
            
            // 重新调整到当前长度的1.2倍
            size_t new_length = static_cast<size_t>(cache_info.current_length * 1.2);
            
            // 调整张量大小
            resize_cache_tensors(cache_info, new_length);
            cache_info.allocated_length = new_length;
            
            std::cout << "Optimized sequence " << pair.first 
                      << ": reduced from " << cache_info.allocated_length 
                      << " to " << new_length << " tokens" << std::endl;
        }
    }
}

template<typename T>
void KVCachePool<T>::resize_cache_tensors(SequenceCacheInfo& cache_info, size_t new_length) {
    // 计算新旧内存大小差值
    size_t old_size = cache_info.allocated_length * config_.num_heads * config_.head_size * sizeof(T);
    size_t new_size = new_length * config_.num_heads * config_.head_size * sizeof(T);
    size_t size_diff = new_size - old_size;
    
    // 创建新的形状
    std::vector<size_t> new_shape = {
        new_length,
        config_.num_heads,
        config_.head_size
    };
    
    // 调整key缓存大小
    for (size_t layer_idx = 0; layer_idx < cache_info.key_caches.size(); ++layer_idx) {
        if (cache_info.key_caches[layer_idx] != nullptr) {
            // 创建新的更大的张量
            Tensor<T>* new_key_cache = new Tensor<T>(new_shape, Device::CUDA, stream_);
            
            // 复制旧数据
            size_t copy_size = std::min(cache_info.allocated_length, new_length);
            
            // 设置复制区域
            std::vector<size_t> copy_shape = {
                copy_size,
                config_.num_heads,
                config_.head_size
            };
            
            // 复制数据 - 修改为使用指针
            new_key_cache->copy_data(cache_info.key_caches[layer_idx]->data_ptr(), 
                                    copy_size * config_.num_heads * config_.head_size * sizeof(T));
            
            // 释放旧缓存内存
            delete cache_info.key_caches[layer_idx];
            
            // 替换旧缓存
            cache_info.key_caches[layer_idx] = new_key_cache;
        }
        if (cache_info.value_caches[layer_idx] != nullptr) {
            // 创建新的更大的张量
            Tensor<T>* new_value_cache = new Tensor<T>(new_shape, Device::CUDA, stream_);
            
            size_t copy_size = std::min(cache_info.allocated_length, new_length);
            
            std::vector<size_t> copy_shape = {
                copy_size,
                config_.num_heads,
                config_.head_size
            };
            
            new_value_cache->copy_data(cache_info.value_caches[layer_idx]->data_ptr(),
                                        copy_size * config_.num_heads * config_.head_size * sizeof(T));
            
            delete cache_info.value_caches[layer_idx];
            
            // 替换旧缓存
            cache_info.value_caches[layer_idx] = new_value_cache;
        }
    }
    
    stats_.total_allocated += size_diff;
}

template class KVCachePool<__nv_bfloat16>;

} // namespace infer