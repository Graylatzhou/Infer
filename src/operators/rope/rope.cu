#include "operators/rope.hpp"

// q or k [seq_len, num_heads, head_dim]
// sin_table [mas_seq_len, head_dim]
// cos_table [mas_seq_len, head_dim]
template <typename T>
__global__ void rope_forward_kernel(
    const T* input,
    const int64_t* position_ids,
    T* output,
    const float* sin_table,
    const float* cos_table,
    const int head_dim,
    int seq_stride,
    int num_heads_stride
) {
    int bx = blockIdx.x; // seq_len
    int by = blockIdx.y; // num_heads
    int tx = threadIdx.x;
    int block_offset = bx * seq_stride + by * num_heads_stride;

    int pos_id = position_ids[bx];
    int table_offset = pos_id * head_dim; //决定table的第几行

    int half_dim = head_dim / 2;
    for (int i = tx; i < half_dim; i += blockDim.x) {
        float sin_value = sin_table[table_offset + i];
        float cos_value = cos_table[table_offset + i];
        if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            float x1 = __bfloat162float(input[block_offset + i]);
            float x2 = __bfloat162float(input[block_offset + i + half_dim]);
            output[block_offset + i] = __float2bfloat16(x1 * cos_value - x2 * sin_value);
            output[block_offset + i + half_dim] = __float2bfloat16(x1 * sin_value + x2 * cos_value);
        } else {
            float x1 = input[block_offset + i];
            float x2 = input[block_offset + i + half_dim];
            output[block_offset + i] = x1 * cos_value - x2 * sin_value;
            output[block_offset + i + half_dim] = x2 * cos_value + x1 * sin_value;
        }
    }
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr, const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr, int rot_offset, int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = VLLM_LDG(cos_ptr + x_index);
    sin = VLLM_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = VLLM_LDG(cos_ptr + x_index / 2);
    sin = VLLM_LDG(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,    // nullptr or
                                   // [batch_size, seq_len, num_kv_heads,
                                   // head_size] or [num_tokens, num_kv_heads,
                                   // head_size]
    const scalar_t* cache_ptr, const int head_size, const int num_heads,
    const int num_kv_heads, const int rot_dim, const int token_idx,
    const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride) {
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        token_idx * query_stride + head_idx * head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int64_t token_head =
          token_idx * key_stride + head_idx * head_stride;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, IS_NEOX>(
          key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
    }
  }
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    scalar_t* __restrict__ query,           // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,  // nullptr or
                                 // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<scalar_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx, query_stride, key_stride, head_stride);
}

void rotary_embedding(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,  // [batch_size, seq_len, num_heads * head_size] or
                           // [num_tokens, num_heads * head_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    std::optional<torch::Tensor> key,
    // null or
    // [batch_size, seq_len, num_kv_heads * head_size] or
    // [num_tokens, num_kv_heads * head_size] or
    // [batch_size, seq_len, num_heads, head_size] or
    // [num_tokens, num_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {
  // num_tokens = batch_size * seq_len
  int64_t num_tokens = positions.numel();
  int positions_ndim = positions.dim();

  // Make sure num_tokens dim is consistent across positions, query, and key
  TORCH_CHECK(
      positions_ndim == 1 || positions_ndim == 2,
      "positions must have shape [num_tokens] or [batch_size, seq_len]");
  if (positions_ndim == 1) {
    TORCH_CHECK(query.size(0) == positions.size(0) &&
                    (!key.has_value() || key->size(0) == positions.size(0)),
                "query, key and positions must have the same number of tokens");
  }
  if (positions_ndim == 2) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (!key.has_value() || key->size(0) == positions.size(0)) &&
            query.size(1) == positions.size(1) &&
            (!key.has_value() || key->size(1) == positions.size(1)),
        "query, key and positions must have the same batch_size and seq_len");
  }

  // Make sure head_size is valid for query and key
  // hidden_size = num_heads * head_size
  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  TORCH_CHECK(query_hidden_size % head_size == 0);
  TORCH_CHECK(key_hidden_size % head_size == 0);

  // Make sure query and key have consistent number of heads
  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key.has_value() ? key_hidden_size / head_size : num_heads;
  TORCH_CHECK(num_heads % num_kv_heads == 0);

  int rot_dim = cos_sin_cache.size(1);
  int seq_dim_idx = positions_ndim - 1;
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;
  // Determine head stride: for [*, heads, head_size] use stride of last dim;
  // for flat [*, heads*head_size], heads blocks are contiguous of size
  // head_size
  int query_ndim = query.dim();
  int64_t head_stride =
      (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (is_neox) {
      rotary_embedding_kernel<__nv_bfloat16, true><<<grid, block, 0, stream>>>(
          reinterpret_cast<int64_t*>(positions.data_ptr()), reinterpret_cast<__nv_bfloat16*>(query.data_ptr()),
          key.has_value() ? reinterpret_cast<__nv_bfloat16*>(key->data_ptr()) : nullptr,
          reinterpret_cast<__nv_bfloat16*>(cos_sin_cache.data_ptr()), rot_dim, query_stride, key_stride,
          head_stride, num_heads, num_kv_heads, head_size);
    } else {
      rotary_embedding_kernel<__nv_bfloat16, false>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<int64_t*>(positions.data_ptr()), reinterpret_cast<__nv_bfloat16*>(query.data_ptr()),
              key.has_value() ? reinterpret_cast<__nv_bfloat16*>(key->data_ptr()) : nullptr,
              reinterpret_cast<__nv_bfloat16*>(cos_sin_cache.data_ptr()), rot_dim, query_stride,
              key_stride, head_stride, num_heads, num_kv_heads, head_size);
    }
}

// void rope_impl(const torch::Tensor& input, const torch::Tensor& position_ids,
//                torch::Tensor& output, const torch::Tensor& sin_table,
//                const torch::Tensor& cos_table) {

//     TORCH_CHECK(input.is_cuda(), "Input tensor must be on a CUDA device");
//     TORCH_CHECK(position_ids.is_cuda(), "Position IDs tensor must be on a CUDA device");
//     TORCH_CHECK(output.is_cuda(), "Output tensor must be on a CUDA device");
//     TORCH_CHECK(sin_table.is_cuda(), "Sin table tensor must be on a CUDA device");
//     TORCH_CHECK(cos_table.is_cuda(), "Cos table tensor must be on a CUDA device");
    
//     // 2. 获取形状信息
//     int seq_len = input.size(0);
//     int num_heads = input.size(1);
//     int head_dim = input.size(2);

//     using T = __nv_bfloat16;

//     // 3. 设置异步执行环境
//     c10::cuda::OptionalCUDAGuard device_guard(input.device());
//     auto stream = at::cuda::getCurrentCUDAStream();

//     // 4. 启动内核
//     dim3 block(head_dim / 2);
//     dim3 grid(seq_len, num_heads);
    
//     rope_forward_kernel<T><<<grid, block, 0, stream>>>(
//         reinterpret_cast<T*>(input.data_ptr()),
//         reinterpret_cast<int64_t*>(position_ids.data_ptr()),
//         reinterpret_cast<T*>(output.data_ptr()),
//         reinterpret_cast<float*>(sin_table.data_ptr()),
//         reinterpret_cast<float*>(cos_table.data_ptr()),
//         head_dim,
//         input.stride(0),
//         input.stride(1)
//     );
// }

// namespace infer {
// template <typename T>
// void RopeOperator<T>::forward(const Tensor<T>* input, Tensor<T>* output, const Tensor<T>* sin_table, const Tensor<T>* cos_table, const Tensor<int32_t>* position_ids) {

//     int seq_len = input->shape()[0];
//     int num_heads = input->shape()[1];
//     int head_dim = input->shape()[2];

//     int seq_stride = input->stride()[0];
//     int num_heads_stride = input->stride()[1];

//     dim3 block(head_dim / 2);
//     dim3 grid(seq_len, num_heads);
    
//     rope_forward_kernel<T, int32_t><<<grid, block, 0, input->getStream()>>>(
//         input->data_ptr(),
//         position_ids->data_ptr(),
//         output->data_ptr(),
//         sin_table->data_ptr(),
//         cos_table->data_ptr(),
//         head_dim,
//         seq_stride,
//         num_heads_stride
//     );
    
// }

// }