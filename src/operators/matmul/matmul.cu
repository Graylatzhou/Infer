#include "operators/matmul.hpp"



#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                           \
    asm volatile(                                                                \
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
        "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
    asm volatile(                                                                \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
        "l"(src), "n"(bytes))

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])

#define LDST128BITSCONST(value) (reinterpret_cast<const float4 *>(&(value))[0])
#define LDST32BITSCONST(value) (reinterpret_cast<const half2 *>(&(value))[0])

#define LDMATRIX_X2_T(R0, R1, addr)                                              \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
                : "=r"(R0), "=r"(R1)                                            \
                : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
    asm volatile(                                                                \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
        : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)            \
    asm volatile(                                                                \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, "  \
        "%4, %5}, {%6, %7}, {%8, %9};\n"                                         \
        : "=r"(RD0), "=r"(RD1)                                                   \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0),  \
        "r"(RC1))


template <int BM=128, int BN=128, int BK=16, bool BLOCK_SWIZZLE=false>
__global__ void gemm_mma_vectorized_kernel(const half *A, const half *B, half *C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    __shared__ half smemA[BM][BK];
    __shared__ half smemB[BK][BN];
    __shared__ half smemC[BM][BN];
    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id % 2;
    int warp_n = warp_id / 2;
    int lane_id = threadIdx.x % 32;
    // load tileA smem idx calculation
    // 128 * 16 layout
    // 16 elements per row
    int load_smem_a_row = threadIdx.x / 2;
    int load_smem_a_col = (threadIdx.x & 1) << 3;
    // load tileB smem idx calculation
    // 16 * 128 layout
    // 128 / 8 = 16 threads per row
    int load_smem_b_row = threadIdx.x / 16;
    int load_smem_b_col = (threadIdx.x & 15) << 3; // 0 - 15 * 8
    int load_gmem_a_row = by * BM + load_smem_a_row;
    int load_gmem_b_col = bx * BN + load_smem_b_col;

    if (load_gmem_a_row >= M || load_gmem_b_col >= N) return;

    uint32_t RC[4][4][2];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
        RC[i][j][0] = 0;
        RC[i][j][1] = 0;
    }
    }
    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    // load tileA
    int load_gmem_a_col = bk * BK + load_smem_a_col;
    int load_gmem_b_row = bk * BK + load_smem_b_row;
    int load_gmem_a_addr = load_gmem_a_row * K + load_gmem_a_col;
    int load_gmem_b_addr = load_gmem_b_row * N + load_gmem_b_col;
    LDST128BITS(smemA[load_smem_a_row][load_smem_a_col]) = LDST128BITSCONST(A[load_gmem_a_addr]);
    LDST128BITS(smemB[load_smem_b_row][load_smem_b_col]) = LDST128BITSCONST(B[load_gmem_b_addr]);
    __syncthreads();
    uint32_t RA[4][4];
    uint32_t RB[4][2];
    // ldmatrixA
    // m 方向2个warp 负责 128行加载 -> 一个warp64行 64 / 4
    // WARP_M = 4 即为RC在M方向的重复次数
    #pragma unroll  
    for (int i = 0; i < 4; i++) { // warp_m * MMA_M * WARP_M
        int lane_load_smem_a_row = warp_m * 16 * 4 + i * 16 + lane_id % 16;
        int lane_load_smem_a_col = (lane_id / 16) * 8;// 0-15 -> 0, 16-31 -> 8
        if (lane_load_smem_a_row >= 128) printf("error");
        uint32_t smemA_addr = __cvta_generic_to_shared(&smemA[lane_load_smem_a_row][lane_load_smem_a_col]);
        LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], smemA_addr);
    }
    // n 方向 4个warp，128列加载 -> 一个warp 32列加载，其中一个warp一次加载8列
    // 所以一个warp要加载4次
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int lane_load_smem_b_row = lane_id % 16; //每个线程提供一行smem的首地址 
        int lane_load_smem_b_col = warp_n * 32 + i * 8; 
        uint32_t smemB_addr = __cvta_generic_to_shared(&smemB[lane_load_smem_b_row][lane_load_smem_b_col]);
        LDMATRIX_X2_T(RB[i][0], RB[i][1], smemB_addr);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
        HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1]);
        }
    }
    __syncthreads();
    }
    // store tileC
    // tileC 128 * 128
    #pragma unroll
    for (int i = 0; i < 4; i++) {

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        // lane_id / 4 -> lane_id 0-32 -> 0-8
        int store_smem_c_row = warp_m * 64 + i * 16 + lane_id / 4; //分为两部分存储，因为RC[2]中的两个元素相隔8行
        int store_smem_c_col = warp_n * 32 + j * 8 + (lane_id % 4) * 2;  // 32bits
        LDST32BITS(smemC[store_smem_c_row + 0][store_smem_c_col]) = LDST32BITS(RC[i][j][0]);
        LDST32BITS(smemC[store_smem_c_row + 8][store_smem_c_col]) = LDST32BITS(RC[i][j][1]);
    }
    }
    for (int tid = threadIdx.x; tid < BM * BN; tid += blockDim.x) {
    int store_smem_c_row = tid / BN;
    int store_smem_c_col = tid % BN;
    int store_gmem_c_row = by * BM + store_smem_c_row;
    int store_gmem_c_col = bx * BN + store_smem_c_col;
    if (store_gmem_c_row < M && store_gmem_c_col < N) {
        C[store_gmem_c_row * N + store_gmem_c_col] += smemC[store_smem_c_row][store_smem_c_col];
    }
    }
}      


template <int BM=128, int BN=128, int BK=16, bool BLOCK_SWIZZLE=false, int KStage=2>
__global__ void gemm_mma_async_vectorized_kernel(const half *A, const half *B, half *C, int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  __shared__ half smemA[KStage][BM][BK];
  __shared__ half smemB[KStage][BK][BN];
  __shared__ half smemC[BM][BN];
  int warp_id = threadIdx.x / 32;
  int warp_m = warp_id % 2;
  int warp_n = warp_id / 2;
  int lane_id = threadIdx.x % 32;
  // load tileA smem idx calculation
  // 128 * 16 layout
  // 16 elements per row
  int load_smem_a_row = threadIdx.x / 2;
  int load_smem_a_col = (threadIdx.x & 1) << 3;
  // load tileB smem idx calculation
  // 16 * 128 layout
  // 128 / 8 = 16 threads per row
  int load_smem_b_row = threadIdx.x / 16;
  int load_smem_b_col = (threadIdx.x & 15) << 3; // 0 - 15 * 8
  int load_gmem_a_row = by * BM + load_smem_a_row;
  int load_gmem_b_col = bx * BN + load_smem_b_col;

  constexpr int sAoffset = BM * BK;
  constexpr int sBoffset = BK * BN;

  if (load_gmem_a_row >= M || load_gmem_b_col >= N) return;

  uint32_t smemA_base_ptr = __cvta_generic_to_shared(&smemA);
  uint32_t smemB_base_ptr = __cvta_generic_to_shared(&smemB);

  uint32_t RC[4][4][2];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }
  // pre-fetch
  #pragma unroll
  for (int turn = 0; turn < KStage - 1; turn++) {
    int load_gmem_a_col = turn * BK + load_smem_a_col;
    int load_gmem_b_row = turn * BK + load_smem_b_row;
    int load_gmem_a_addr = load_gmem_a_row * K + load_gmem_a_col;
    int load_gmem_b_addr = load_gmem_b_row * N + load_gmem_b_col;
    uint32_t smemA_addr = smemA_base_ptr + (turn * sAoffset + load_smem_a_row * BK + load_smem_a_col) * sizeof(half);
    CP_ASYNC_CG(smemA_addr, &A[load_gmem_a_addr], 16); //bytes, CG bypass L1
    uint32_t smemB_addr = smemB_base_ptr + (turn * sBoffset + load_smem_b_row * BN + load_smem_b_col) * sizeof(half);
    CP_ASYNC_CG(smemB_addr, &B[load_gmem_b_addr], 16); //bytes, CG bypass L1
    CP_ASYNC_COMMIT_GROUP();
  }
  CP_ASYNC_WAIT_GROUP(KStage - 2); // 等待到还剩KStage - 2个操作未完成
  __syncthreads();
  for (int bk = KStage - 1; bk < (K + BK - 1) / BK; bk++) {
    // load tileA
    int smem_select_next = bk % KStage; // load
    int smem_select = (bk + 1) % KStage; // calculate
    int load_gmem_a_col = bk * BK + load_smem_a_col;
    int load_gmem_b_row = bk * BK + load_smem_b_row;
    int load_gmem_a_addr = load_gmem_a_row * K + load_gmem_a_col;
    int load_gmem_b_addr = load_gmem_b_row * N + load_gmem_b_col;
    uint32_t smemA_addr = smemA_base_ptr + (smem_select_next * sAoffset + load_smem_a_row * BK + load_smem_a_col) * sizeof(half);
    uint32_t smemB_addr = smemB_base_ptr + (smem_select_next * sBoffset + load_smem_b_row * BN + load_smem_b_col) * sizeof(half);
    CP_ASYNC_CG(smemA_addr, &A[load_gmem_a_addr], 16); //bytes, CG bypass L1
    CP_ASYNC_CG(smemB_addr, &B[load_gmem_b_addr], 16); //bytes, CG bypass L1
    CP_ASYNC_COMMIT_GROUP();
    uint32_t RA[4][4];
    uint32_t RB[4][2];
    // ldmatrixA
    // m 方向2个warp 负责 128行加载 -> 一个warp64行 64 / 4
    // WARP_M = 4 即为RC在M方向的重复次数
    #pragma unroll  
    for (int i = 0; i < 4; i++) { // warp_m * MMA_M * WARP_M
      int lane_load_smem_a_row = warp_m * 16 * 4 + i * 16 + lane_id % 16;
      int lane_load_smem_a_col = (lane_id / 16) * 8;// 0-15 -> 0, 16-31 -> 8
      uint32_t smemA_addr = __cvta_generic_to_shared(&smemA[smem_select][lane_load_smem_a_row][lane_load_smem_a_col]);
      LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], smemA_addr);
    }
    // n 方向 4个warp，128列加载 -> 一个warp 32列加载，其中一个warp一次加载8列
    // 所以一个warp要加载4次
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int lane_load_smem_b_row = lane_id % 16; //每个线程提供一行smem的首地址 
      int lane_load_smem_b_col = warp_n * 32 + i * 8; 
      uint32_t smemB_addr = __cvta_generic_to_shared(&smemB[smem_select][lane_load_smem_b_row][lane_load_smem_b_col]);
      LDMATRIX_X2_T(RB[i][0], RB[i][1], smemB_addr);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1]);
      }
    }
    CP_ASYNC_WAIT_GROUP(KStage - 2);
    __syncthreads();
  }
  //calculate buffer idx 1
  uint32_t RA[4][4];
  uint32_t RB[4][2];
  #pragma unroll  
  for (int i = 0; i < 4; i++) { // warp_m * MMA_M * WARP_M
    int lane_load_smem_a_row = warp_m * 16 * 4 + i * 16 + lane_id % 16;
    int lane_load_smem_a_col = (lane_id / 16) * 8;// 0-15 -> 0, 16-31 -> 8
    uint32_t smemA_addr = __cvta_generic_to_shared(&smemA[1][lane_load_smem_a_row][lane_load_smem_a_col]);
    LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], smemA_addr);
  }
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    int lane_load_smem_b_row = lane_id % 16; //每个线程提供一行smem的首地址 
    int lane_load_smem_b_col = warp_n * 32 + i * 8; 
    uint32_t smemB_addr = __cvta_generic_to_shared(&smemB[1][lane_load_smem_b_row][lane_load_smem_b_col]);
    LDMATRIX_X2_T(RB[i][0], RB[i][1], smemB_addr);
  }
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1]);
    }
  }
  // store tileC
  // tileC 128 * 128
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      // lane_id / 4 -> lane_id 0-32 -> 0-8
      int store_smem_c_row = warp_m * 64 + i * 16 + lane_id / 4; //分为两部分存储，因为RC[2]中的两个元素相隔8行
      int store_smem_c_col = warp_n * 32 + j * 8 + (lane_id % 4) * 2;  // 32bits
      LDST32BITS(smemC[store_smem_c_row + 0][store_smem_c_col]) = LDST32BITS(RC[i][j][0]);
      LDST32BITS(smemC[store_smem_c_row + 8][store_smem_c_col]) = LDST32BITS(RC[i][j][1]);
    }
  }
  for (int tid = threadIdx.x; tid < BM * BN; tid += blockDim.x) {
    int store_smem_c_row = tid / BN;
    int store_smem_c_col = tid % BN;
    int store_gmem_c_row = by * BM + store_smem_c_row;
    int store_gmem_c_col = bx * BN + store_smem_c_col;
    if (store_gmem_c_row < M && store_gmem_c_col < N) {
      C[store_gmem_c_row * N + store_gmem_c_col] += smemC[store_smem_c_row][store_smem_c_col];
    }
  }
}

template <typename T, int BM, int BN, int BK, int kStage, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB, typename SmemLayoutA,
          typename SmemLayoutB, typename SmemLayoutC, typename S2RCopyAtomA,
          typename S2RCopyAtomB, typename R2SCopyAtomC, typename S2GCopyAtomC,
          typename S2GCopyC, const bool BlockSwizzle>
__global__ void cutlass_mma_stages_block_swizzle_tn_kernel(const void *Aptr, const void *Bptr,
                                                              void *Dptr, int m,
                                                              int n, int k) {
  using namespace cute;
  // Initilize shared memory
  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  // Initilize thread block
  int idx = threadIdx.x;
  // BlockSwizzle 0/1 control use block swizzle or not.
  int ix = ((int)BlockSwizzle) * blockIdx.z * gridDim.x + blockIdx.x;
  int iy = blockIdx.y;

  if (iy * BM >= m || ix * BN >= n)
    return;

  // use Tensor notation to represent device pointer + dimension
  Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));

  // slice the tensor to small one which is used for current thread block.
  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}),
                         make_coord(iy, _)); // (BM, BK, num_tile_k)
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}),
                         make_coord(ix, _)); // (BN, BK, num_tile_k)
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}),
                         make_coord(iy, ix)); // (BM, BN)

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{}); // (BM, BK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{}); // (BN, BK, kStage)

  // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tCgD = thr_mma.partition_C(gD); // (MMA,MMA_M, MMA_N)

  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)
  clear(tCrD);

  // from global memory to shared memory
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy =
      g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, num_tile_k)
  auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, kStage)
#ifdef CUTE_HGEMM_DEBUG
  if (thread0()) {
    print("\npartition_S(tAgA_copy): \n");
    print(tAgA_copy);
    print("\n");
    print("\nThrCopy(g2s_thr_copy_a): \n");
    print(g2s_thr_copy_a);
    print("\n");
  }
#endif

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy =
      g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, num_tile_k)
  auto tBsB_copy =
      g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K, kStage)

  // from shared memory to register, use tiled_mma to generate tiled_copy
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K, kStage)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

  /* PREFETCH */
  // submit kStage - 1 tile
  // gmem -> shm
  int itile_to_read = 0;
  int ismem_read = 0;
  int ismem_write = 0;

#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
               tAsA_copy(_, _, _, istage));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
               tBsB_copy(_, _, _, istage));
    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  // wait one submitted gmem->smem done
  cp_async_wait<kStage - 2>();
  __syncthreads();

  int ik = 0;
  // smem -> reg
  // tAsA: (CPY, CPY_M, CPY_K, kStage) tCrA_view: (CPY, CPY_M, CPY_K)
  cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  // loop over k: i. load tile, ii. mma
  int ntile = k / BK;
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
    int nk = size<2>(tCrA); // (MMA, MMA_M, MMA_K)

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      int ik_next = (ik + 1) % nk;

      if (ik == nk - 1) {
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage;
      }

      // shm -> reg s[itile][ik + 1] -> r[ik + 1]
      // tAsA: (CPY, CPY_M, CPY_K, kStage), tCrA_view: (CPY, CPY_M, CPY_K)
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                 tCrA_view(_, _, ik_next));
      // tBsB: (CPY, CPY_M, CPY_K, kStage), tCrB_view: (CPY, CPY_M, CPY_K)
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));

      if (ik == 0) {
        if (itile_to_read < ntile) {
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }

        cp_async_fence();
      }

      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    } // for ik
  }

  // use less shared memory as a scratchpad tile to use large wide instuction
  // Dreg -> shm -> reg -> global
  auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);  // (CPY, CPY_M, CPY_N)
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, _1, _1, pipe)

  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC); // (CPY, _1, _1, pipe)
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD); // (CPY, CPY_M, CPY_N)

  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g); // (CPY_, CPY_MN)
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s); // (CPY_, CPY_MN)

  int step = size<3>(tCsC_r2s); // pipe
#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
// reg -> shm
#pragma unroll
    for (int j = 0; j < step; ++j) {
      // we add a temp tensor to cope with accumulator and output data type
      // difference
      auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
      cute::copy(tCrC_r2sx(_, i + j), t);

      cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();

#pragma unroll
    // shm -> global
    for (int j = 0; j < step; ++j) {
      cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }
    __syncthreads();
  } // end for
}

// For torch binding, need dynamic block swizzle stride
template <typename T, const int Stages = 2, const bool BlockSwizzle = false>
void launch_hgemm_mma_stages_block_swizzle_tn_cute(const void *a, const void *b, void *c, int M,
                                                   int N, int K,
                                                   int swizzle_stride,
                                                   cudaStream_t stream) {
  using namespace cute;

  auto BM = Int<128>{};
  auto BN = Int<256>{};
  auto BK = Int<32>{};
  auto KStage = Int<Stages>{};       // default 2
  auto kSmemLayoutCBatch = Int<4>{}; // namely, stages.

  // Define the smem layouts, Swizzle<3, 3, 3> and
  // Swizzle<2, 3, 3> will get the same results.
  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, make_layout(make_shape(Int<8>{}, Int<BK>{}),
                                      make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{}))); // (m,n) -> smem_idx
#ifdef CUTE_HGEMM_DEBUG
  print("SmemLayoutA: ");
  print(SmemLayoutA{});
  print("\n");
  print("SmemLayoutB: ");
  print(SmemLayoutB{});
  print("\n");
  print("SmemLayoutB: ");
  print(SmemLayoutB{});
  print("\n");
  print("SmemLayoutAtom A&B Latex: \n");
  print_latex(SmemLayoutAtom{});
  print("\n");
#endif

  // mma
  using mma_op = SM80_16x8x16_F32BF16BF16F32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  static constexpr int kMmaEURepeatM = 2; // MMA repeat 2 times across M
  static constexpr int kMmaEURepeatN = 2; // MMA repeat 2 times across N
  static constexpr int kMmaEURepeatK = 1; // MMA no repeat across K

  using mma_atom_shape = mma_traits::Shape_MNK; // M,N,K 16,8,16
  static constexpr int kMmaPM =
      1 * kMmaEURepeatM * get<0>(mma_atom_shape{}); // 1*2*16=32
  static constexpr int kMmaPN =
      2 * kMmaEURepeatN * get<1>(mma_atom_shape{}); // 2*2*8 =32
  static constexpr int kMmaPK =
      1 * kMmaEURepeatK * get<2>(mma_atom_shape{}); // 1*1*16=16
  // TiledMMA, more threads, MMAThrLayout(2,2,1), 4 MMA = 4 warps = 32x4
  // threads.
  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  // TiledMMA, more values, Permutations(32,32,16)
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
#ifdef CUTE_HGEMM_DEBUG
  print("MMA: ");
  print(MMA{});
  print("\n");
  print("MMA Latex: \n");
  print_latex(MMA{});
  print("\n");
#endif

  // copy from global memory to shared memory
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
  // Make TiledCopy according to ThrLayout and ValLayout.
  // 32x4 threads, each thread load 1x8 values (128 bits) once ?
  //   Produce a TiledCopy from logical thread and values layouts.
  // The thread and value layouts map coordinates to thr_idx and val_idx.
  //   The product of these layouts is taken to produce the TV layout and the
  //   Tiler.
  // Useful when threads and values need very specific mappings onto coordinates
  //   in the target tensors.
  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<4>{}), // Thr layout 32x4 k-major
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{})))); // Val layout 1x8
  using G2SCopyB = G2SCopyA;
#ifdef CUTE_HGEMM_DEBUG
  print("G2SCopyA: ");
  print(G2SCopyA{});
  print("\n");
  print("G2SCopyB: ");
  print(G2SCopyB{});
  print("\n");
  print("G2SCopyA Latex: \n");
  print_latex(G2SCopyA{});
  print("\n");
  print("G2SCopyB Latex: \n");
  print_latex(G2SCopyB{});
  print("\n");
#endif
  // copy from shared memory to register
  // use mma tiled ,so no tiled here
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  // epilogue: register to global via shared memory
  // Swizzle<3, 3, 3>=BxMxS=(2^3)*(2^3)*(2^3)=512 values=1024 bytes.
  // reference: https://zhuanlan.zhihu.com/p/671419093
  using SmemLayoutAtomC = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}), // 32*32
                  make_stride(Int<kMmaPN>{}, Int<1>{}))));
  // kSmemLayoutCBatch=4, 32x32x4=4096 values=8192 bytes
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

  static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
                    size(SmemLayoutC{}),
                "C shared memory request is large than A's one pipe");
#ifdef CUTE_HGEMM_DEBUG
  print(SmemLayoutC{});
  print("\n");
  static constexpr int tmp_sizeC = size(SmemLayoutC{});
  static constexpr int tmp_sizeA_0 = size<0>(SmemLayoutA{});
  static constexpr int tmp_sizeA_1 = size<1>(SmemLayoutA{});
  static constexpr int tmp_sizeA = tmp_sizeA_0 * tmp_sizeA_1;
  print("size SmemLayoutC: %d", tmp_sizeC);
  print("\n");
  print("size SmemLayoutA: %d", tmp_sizeA);
  print("\n");
  print("size 0 SmemLayoutA: %d", tmp_sizeA_0);
  print("\n");
  print("size 1 SmemLayoutA: %d", tmp_sizeA_1);
  print("\n");
#endif

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;

  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC =
      decltype(make_tiled_copy(S2GCopyAtomC{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  int BX = (N + BN - 1) / BN;
  int BY = (M + BM - 1) / BM;

  int BZ = BlockSwizzle ? (N + (swizzle_stride)-1) / (swizzle_stride) : 1;
  BX = BlockSwizzle ? (BX + BZ - 1) / BZ : BX;

  dim3 block(size(MMA{}));
  dim3 grid(BX, BY, BZ);

  // C_shm is shared with A_shm and B_shm
  // we don't allocate new smem for C_shm.
  // (128 * 32 * 2) * 2 + (256 * 32 * 2) * 2 = 49152 bytes, stages=2
  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);

  int shm_size = kShmSize;
#ifdef CUTE_HGEMM_DEBUG
  print("shm_size: %d bytes, shm_size_AB: %d bytes, shm_size_C: %d bytes\n",
        shm_size, shm_size_AB * (int)sizeof(T), shm_size_C * (int)sizeof(T));
#endif

  cudaFuncSetAttribute(
      cutlass_mma_stages_block_swizzle_tn_kernel<
          T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB, SmemLayoutA,
          SmemLayoutB, SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC,
          S2GCopyAtomC, S2GCopyC, BlockSwizzle>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  cutlass_mma_stages_block_swizzle_tn_kernel<
      T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
      SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC,
      S2GCopyC, BlockSwizzle><<<grid, block, shm_size, stream>>>(a, b, c, M, N, K);
}

__global__ void print(const half* data) {
    printf("half data: %f\n", __half2float(data[0]));
    printf("half data: %f\n", __half2float(data[1]));
}
namespace infer {
template <>
void MatMulOperator<__nv_bfloat16>::forward(const Tensor<__nv_bfloat16>* A, const Tensor<__nv_bfloat16>* B, Tensor<__nv_bfloat16>* output, Tensor<__nv_bfloat16>* bias) {
    if (A->ndim() != 2 || B->ndim() != 2) {
        throw std::runtime_error("Both input tensors must be 2D matrices.");
    }
    int M = A->shape()[0];
    int N = B->shape()[0];
    int K = A->shape()[1];
    launch_hgemm_mma_stages_block_swizzle_tn_cute<cute::bfloat16_t, 3, false>(
        A->void_ptr(), B->void_ptr(), output->void_ptr(), M, N, K, 1024, A->getStream());

}
template <typename T>
MatMulOperator<T>::MatMulOperator() {
    // 初始化 cuBLAS 句柄
    cublasCreate(&handle_);
}

template <typename T>
MatMulOperator<T>::~MatMulOperator() {
    if (handle_) {
        cublasDestroy(handle_);
        handle_ = nullptr;
    }
}


template <>
void MatMulOperator<float>::forward(const Tensor<float>* A, const Tensor<float>* B, Tensor<float>* output, Tensor<float>* bias) {
    if (A->ndim() != 2 || B->ndim() != 2) {
        throw std::runtime_error("Both input tensors must be 2D matrices.");
    }

    int m = A->shape()[0];
    int n = B->shape()[1];
    int k = A->shape()[1];
    
    cublasSetStream(handle_, A->getStream());
    cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH);
    
  
    static float alpha = 1.0;
    static float beta = 0.0;
  
    cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B->data_ptr(), CUDA_R_32F,
                 n, A->data_ptr(), CUDA_R_32F, k, &beta, output->data_ptr(), CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
    
}

}

