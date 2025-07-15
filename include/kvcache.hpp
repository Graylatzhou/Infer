#pragma once

#include "memorypool.hpp"
#include "tensor.hpp"
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace infer {

struct KVCachePoolConfig {
    size_t block_size;            // 块大小 (token数)
    size_t head_size;             // 头大小
    size_t num_heads;             // 头数量
    size_t num_layers;            // 层数
    bool use_mla;                 // 是否使用MLA (多查询注意力)
    
    KVCachePoolConfig(
        size_t blk_size = 512,
        size_t h_size = 128,
        size_t n_heads = 32,
        size_t n_layers = 32,
        bool mla = false
    ) : block_size(blk_size),
        head_size(h_size),
        num_heads(n_heads),
        num_layers(n_layers),
        use_mla(mla) {}
};

template<typename T>
class KVCachePool {
public:
    KVCachePool(const KVCachePoolConfig& config, cudaStream_t stream = nullptr);
    ~KVCachePool();
    
    // 禁用复制
    KVCachePool(const KVCachePool&) = delete;
    KVCachePool& operator=(const KVCachePool&) = delete;
    
    // 允许移动
    KVCachePool(KVCachePool&&) = default;
    KVCachePool& operator=(KVCachePool&&) = default;
    
    // 获取序列的KV缓存
    Tensor<T> get_key_cache(size_t sequence_id, size_t layer_idx);
    Tensor<T> get_value_cache(size_t sequence_id, size_t layer_idx);
    
    // 扩展序列的KV缓存
    void extend_sequence(size_t sequence_id, size_t new_length);
    
    // 释放序列的KV缓存
    void free_sequence(size_t sequence_id);
    
    // 预分配序列缓存
    void prefetch_sequence(size_t sequence_id, size_t initial_length);
    
    // 获取内存使用统计
    void print_memory_stats() const;
    
    // 优化内存使用
    void optimize_memory();

private:
    // 序列缓存信息
    struct SequenceCacheInfo {
        std::vector<Tensor<T>> key_caches;   // 每层的key缓存
        std::vector<Tensor<T>> value_caches; // 每层的value缓存
        size_t current_length;               // 当前序列长度
        size_t allocated_length;             // 已分配长度
    };
    
    // 序列缓存映射表
    std::unordered_map<size_t, SequenceCacheInfo> sequence_caches_;
    
    // 配置
    KVCachePoolConfig config_;
    
    // CUDA流
    cudaStream_t stream_;
    
    // 统计信息
    struct MemoryStats {
        size_t total_allocated;   // 总分配内存
        size_t active_sequences;  // 活跃序列数
        size_t max_sequence_len;  // 最长序列长度
    };
    
    mutable MemoryStats stats_;
    
    // 线程安全
    mutable std::mutex mutex_;
    
    // 内部方法
    void resize_cache_tensors(SequenceCacheInfo& cache_info, size_t new_length);
};

} // namespace infer