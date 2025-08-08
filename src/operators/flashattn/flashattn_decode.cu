#include <torch/extension.h>
/*
decode阶段
q seq_len = 1
k seq_len = kvcache_lenth
v seq_len = kvcache_length
现在只考虑batch_size = 1的情况
*/


// void flash_attn_decode(const Tensor& Q, const Tensor& K, const Tensor& V, // kvcache即为这里的KV，我需要在这里接入blocktable
//                        bool is_causal, Tensor& block_table, Tensor& O) {

// }