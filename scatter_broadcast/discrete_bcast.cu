#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include "cuda_sm_kernel.h"

// -------------------- utils --------------------
#define CUDA_CHECK(cmd) do {                                 \
  cudaError_t e = cmd;                                       \
  if (e != cudaSuccess)                                      \
    throw std::runtime_error(cudaGetErrorString(e));         \
} while(0)

#define NCCL_CHECK(cmd) do {                                 \
  ncclResult_t r = cmd;                                      \
  if (r != ncclSuccess)                                      \
    throw std::runtime_error(ncclGetErrorString(r));         \
} while(0)

// -------------------- NCCL comm handle --------------------
struct CommHandle {
  ncclComm_t comm;
  int rank, world;

  // pipeline resources
  bool pipe_inited = false;
  cudaStream_t copy_stream = nullptr;
  cudaStream_t nccl_stream = nullptr;
  cudaEvent_t gather_done[2];
  cudaEvent_t bcast_done[2];
  cudaEvent_t scatter_done[2];
  void* staging[2] = {nullptr, nullptr};
  size_t staging_bytes = 0;
};

static void ensure_pipeline_resources(CommHandle* h, size_t need_bytes) {
  if (!h->pipe_inited) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&h->copy_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&h->nccl_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&h->gather_done[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&h->gather_done[1], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&h->bcast_done[0],  cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&h->bcast_done[1],  cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&h->scatter_done[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&h->scatter_done[1], cudaEventDisableTiming));
    h->pipe_inited = true;
  }

  if (h->staging_bytes < need_bytes) {
    // 释放旧 buffer
    if (h->staging[0]) CUDA_CHECK(cudaFree(h->staging[0]));
    if (h->staging[1]) CUDA_CHECK(cudaFree(h->staging[1]));
    // 重新分配双缓冲
    CUDA_CHECK(cudaMalloc(&h->staging[0], need_bytes));
    CUDA_CHECK(cudaMalloc(&h->staging[1], need_bytes));
    h->staging_bytes = need_bytes;
  }
}

static void ensure_resources(CommHandle* h, size_t need_bytes) {
  if (!h->pipe_inited) {
    h->pipe_inited = true;
  }

  if (h->staging_bytes < need_bytes) {
    // 释放旧 buffer
    if (h->staging[0]) CUDA_CHECK(cudaFree(h->staging[0]));
    // 重新分配双缓冲
    CUDA_CHECK(cudaMalloc(&h->staging[0], need_bytes));
    h->staging_bytes = need_bytes;
  }
}

static std::vector<CommHandle*> g_handles; // very simple handle table

static int store_handle(CommHandle* h) {
  g_handles.push_back(h);
  return (int)g_handles.size(); // handle id starts at 1
}

static CommHandle* get_handle(int hid) {
  if (hid <= 0 || hid > (int)g_handles.size() || g_handles[hid-1] == nullptr)
    throw std::runtime_error("Invalid NCCL handle");
  return g_handles[hid-1];
}

static void destroy_handle(int hid) {
  CommHandle* h = get_handle(hid);
  NCCL_CHECK(ncclCommDestroy(h->comm));
  delete h;
  g_handles[hid-1] = nullptr;
}

// -------------------- gather/scatter kernels --------------------
// pointers: uint64 device pointers
__global__ void gather_kernel(
    const uint64_t* __restrict__ src_ptrs,
    const int64_t* __restrict__ lens,
    const int64_t* __restrict__ byte_offsets, // prefix sum offsets in bytes
    int n,
    uint8_t* __restrict__ staging,
    int elem_size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // total bytes unknown here; loop per segment
  for (int i = 0; i < n; ++i) {
    int64_t bytes = lens[i] * (int64_t)elem_size;
    int64_t base = byte_offsets[i];
    uint8_t* dst = staging + base;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(src_ptrs[i]);
    // copy bytes in a strided way
    for (int64_t j = tid; j < bytes; j += (int64_t)gridDim.x * blockDim.x) {
      dst[j] = src[j];
    }
  }
}

__global__ void scatter_kernel(
    const uint64_t* __restrict__ dst_ptrs,
    const int64_t* __restrict__ lens,
    const int64_t* __restrict__ byte_offsets,
    int n,
    const uint8_t* __restrict__ staging,
    int elem_size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < n; ++i) {
    int64_t bytes = lens[i] * (int64_t)elem_size;
    int64_t base = byte_offsets[i];
    const uint8_t* src = staging + base;
    uint8_t* dst = reinterpret_cast<uint8_t*>(dst_ptrs[i]);
    for (int64_t j = tid; j < bytes; j += (int64_t)gridDim.x * blockDim.x) {
      dst[j] = src[j];
    }
  }
}

// -------------------- exposed functions --------------------
int create_nccl_comm(int world_size, int rank, py::bytes unique_id_bytes) {
  std::string uid_str = unique_id_bytes; // should be sizeof(ncclUniqueId)
  if (uid_str.size() != sizeof(ncclUniqueId))
    throw std::runtime_error("unique_id bytes size mismatch");

  ncclUniqueId uid;
  std::memcpy(&uid, uid_str.data(), sizeof(ncclUniqueId));

  auto* h = new CommHandle();
  h->world = world_size;
  h->rank = rank;
  NCCL_CHECK(ncclCommInitRank(&h->comm, world_size, uid, rank));
  return store_handle(h);
}

void destroy_nccl_comm(int handle) {
  destroy_handle(handle);
}

// void broadcast_discrete(
//     int handle,
//     torch::Tensor ptrs_u64_cpu,   // CPU tensor (uint64) holding device pointers
//     torch::Tensor lens_i64_cpu,   // CPU tensor (int64) holding lengths (elements)
//     int elem_size,
//     int root,
//     uint64_t stream_ptr /*0 => current*/)
// {
//   auto* h = get_handle(handle);
//   if (!ptrs_u64_cpu.is_cpu() || !lens_i64_cpu.is_cpu())
//     throw std::runtime_error("ptrs/lens must be CPU tensors (from numpy)");

//   if (ptrs_u64_cpu.scalar_type() != torch::kUInt64)
//     throw std::runtime_error("ptrs must be uint64");
//   if (lens_i64_cpu.scalar_type() != torch::kInt64)
//     throw std::runtime_error("lens must be int64");

//   int64_t n = ptrs_u64_cpu.numel();
//   if (lens_i64_cpu.numel() != n) throw std::runtime_error("ptrs/lens size mismatch");

//   auto ptrs_acc = ptrs_u64_cpu.contiguous();
//   auto lens_acc = lens_i64_cpu.contiguous();

//   // compute byte offsets on CPU (prefix sum)
//   std::vector<int64_t> offsets(n);
//   int64_t total_bytes = 0;
//   auto lens_ptr = (int64_t*)lens_acc.data_ptr<int64_t>();
//   for (int64_t i = 0; i < n; ++i) {
//     offsets[i] = total_bytes;
//     total_bytes += lens_ptr[i] * (int64_t)elem_size;
//   }

//   // copy ptrs + lens + offsets to device
//   auto opts_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCUDA);
//   auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

//   torch::Tensor d_ptrs = torch::empty({n}, opts_u64);
//   torch::Tensor d_lens = torch::empty({n}, opts_i64);
//   torch::Tensor d_offs = torch::empty({n}, opts_i64);

//   int dev = -1;
//   CUDA_CHECK(cudaGetDevice(&dev));
//   cudaStream_t stream =
//     stream_ptr ? reinterpret_cast<cudaStream_t>(stream_ptr)
//                : c10::cuda::getDefaultCUDAStream(dev);

//   CUDA_CHECK(cudaMemcpyAsync(d_ptrs.data_ptr(), ptrs_acc.data_ptr(), n * sizeof(uint64_t),
//                              cudaMemcpyHostToDevice, stream));
//   CUDA_CHECK(cudaMemcpyAsync(d_lens.data_ptr(), lens_acc.data_ptr(), n * sizeof(int64_t),
//                              cudaMemcpyHostToDevice, stream));
//   CUDA_CHECK(cudaMemcpyAsync(d_offs.data_ptr(), offsets.data(), n * sizeof(int64_t),
//                              cudaMemcpyHostToDevice, stream));

//   // staging buffer on device
//   torch::Tensor staging = torch::empty({total_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

//   // gather -> ncclBroadcast -> scatter
//   int threads = 256;
//   int blocks = 120; // simple fixed, can tune
//   gather_kernel<<<blocks, threads, 0, stream>>>(
//       (const uint64_t*)d_ptrs.data_ptr<uint64_t>(),
//       (const int64_t*)d_lens.data_ptr<int64_t>(),
//       (const int64_t*)d_offs.data_ptr<int64_t>(),
//       (int)n,
//       (uint8_t*)staging.data_ptr<uint8_t>(),
//       elem_size);

//   NCCL_CHECK(ncclBroadcast(
//       staging.data_ptr(),
//       staging.data_ptr(),
//       (size_t)total_bytes,
//       ncclUint8,
//       root,
//       h->comm,
//       stream));

//   scatter_kernel<<<blocks, threads, 0, stream>>>(
//       (const uint64_t*)d_ptrs.data_ptr<uint64_t>(),
//       (const int64_t*)d_lens.data_ptr<int64_t>(),
//       (const int64_t*)d_offs.data_ptr<int64_t>(),
//       (int)n,
//       (const uint8_t*)staging.data_ptr<uint8_t>(),
//       elem_size);
// }

void broadcast_discrete(
    int handle,
    torch::Tensor d_ptrs_u64,   // CUDA uint64 [n]
    int64_t n_blocks,
    int64_t block_bytes,
    int root,
    uint64_t stream_ptr)
{
  auto* h = get_handle(handle);

  if (!d_ptrs_u64.is_cuda()) throw std::runtime_error("d_ptrs_u64 must be CUDA tensor");
  if (d_ptrs_u64.scalar_type() != torch::kUInt64) throw std::runtime_error("d_ptrs_u64 must be uint64");
  if (d_ptrs_u64.numel() != n_blocks) throw std::runtime_error("n_blocks mismatch");
  if (block_bytes <= 0) throw std::runtime_error("block_bytes <= 0");

  // 你的 copy kernel 要求 32B 对齐（至少建议）
  if ((block_bytes % 32) != 0) throw std::runtime_error("block_bytes must be multiple of 32");

  // reinterpret to void** (device pointer array)
  auto d_ptrs_u64_contig = d_ptrs_u64.contiguous();
  uint64_t* p_u64 = (uint64_t*)d_ptrs_u64_contig.data_ptr<uint64_t>();
  void** d_ptrs = (void**)p_u64;
  size_t staging_bytes = (size_t)(n_blocks * block_bytes);
  // ensure internal streams/buffers
  ensure_resources(h, staging_bytes);
  void* staging = h->staging[0];

  int dev = -1;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaStream_t stream =
    stream_ptr ? reinterpret_cast<cudaStream_t>(stream_ptr)
               : c10::cuda::getDefaultCUDAStream(dev);
  if (h->rank == root) {
    auto e = UC::Trans::CudaSMCopyAsync((void**)d_ptrs, staging,
                                        (size_t)block_bytes, (size_t)n_blocks, stream);
    if (e != cudaSuccess) throw std::runtime_error("gather failed");
  }

  NCCL_CHECK(ncclBroadcast(
    staging,
    staging,
    staging_bytes,
    ncclUint8,
    root,
    h->comm,
    stream));
  
  // --- 3) scatter (non-root) on copy_stream, must wait for bcast_done
  if (h->rank != root) {
    auto e2 = UC::Trans::CudaSMCopyAsync(staging, (void**)d_ptrs,
                                          (size_t)block_bytes, (size_t)n_blocks, stream);
    if (e2 != cudaSuccess) throw std::runtime_error("scatter failed");
  }
}

void broadcast_discrete_pipelined(
    int handle,
    torch::Tensor d_ptrs_u64,   // CUDA uint64 [n]
    int64_t n_blocks,
    int64_t block_bytes,
    int64_t max_chunk_bytes,
    int root)
{
  auto* h = get_handle(handle);

  if (!d_ptrs_u64.is_cuda()) throw std::runtime_error("d_ptrs_u64 must be CUDA tensor");
  if (d_ptrs_u64.scalar_type() != torch::kUInt64) throw std::runtime_error("d_ptrs_u64 must be uint64");
  if (d_ptrs_u64.numel() != n_blocks) throw std::runtime_error("n_blocks mismatch");
  if (block_bytes <= 0) throw std::runtime_error("block_bytes <= 0");
  if (max_chunk_bytes <= 0) throw std::runtime_error("max_chunk_bytes <= 0");

  // 你的 copy kernel 要求 32B 对齐（至少建议）
  if ((block_bytes % 32) != 0) throw std::runtime_error("block_bytes must be multiple of 32");

  // reinterpret to void** (device pointer array)
  auto d_ptrs_u64_contig = d_ptrs_u64.contiguous();
  uint64_t* p_u64 = (uint64_t*)d_ptrs_u64_contig.data_ptr<uint64_t>();
  void** d_ptrs = (void**)p_u64;

  // chunk config
  int64_t blocks_per_chunk = std::max<int64_t>(1, max_chunk_bytes / block_bytes);
  int64_t num_chunks = (n_blocks + blocks_per_chunk - 1) / blocks_per_chunk;
  size_t staging_bytes = (size_t)(blocks_per_chunk * block_bytes);

  // ensure internal streams/buffers
  ensure_pipeline_resources(h, staging_bytes);

  cudaStream_t copy_stream = h->copy_stream;
  cudaStream_t nccl_stream = h->nccl_stream;

  // init events to "done" (avoid first wait issues)
  CUDA_CHECK(cudaEventRecord(h->bcast_done[0], nccl_stream));
  CUDA_CHECK(cudaEventRecord(h->bcast_done[1], nccl_stream));

  for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
    int i = (int)(chunk_id & 1);

    int64_t start = chunk_id * blocks_per_chunk;
    int64_t cnt   = std::min<int64_t>(blocks_per_chunk, n_blocks - start);
    size_t bytes  = (size_t)(cnt * block_bytes);

    void* staging = h->staging[i];
    void** list_ptrs = d_ptrs + start;

    if (h->rank == root) {
      CUDA_CHECK(cudaStreamWaitEvent(copy_stream, h->bcast_done[i], 0));
    } else {
      // 2) non-root: ncclBroadcast 会写 staging[i]，必须等上一轮该 buffer 的 scatter 读完
      CUDA_CHECK(cudaStreamWaitEvent(nccl_stream, h->scatter_done[i], 0));
    }

    // --- 1) gather (root only) on copy_stream
    if (h->rank == root) {
      auto e = UC::Trans::CudaSMCopyAsync((void**)list_ptrs, staging,
                                          (size_t)block_bytes, (size_t)cnt, copy_stream);
      if (e != cudaSuccess) throw std::runtime_error("gather failed");

      CUDA_CHECK(cudaEventRecord(h->gather_done[i], copy_stream));
      CUDA_CHECK(cudaStreamWaitEvent(nccl_stream, h->gather_done[i], 0));
    }

    // --- 2) nccl broadcast on nccl_stream
    NCCL_CHECK(ncclBroadcast(staging, staging, bytes, ncclUint8, root, h->comm, nccl_stream));
    CUDA_CHECK(cudaEventRecord(h->bcast_done[i], nccl_stream));

    // --- 3) scatter (non-root) on copy_stream, must wait for bcast_done
    if (h->rank != root) {
      CUDA_CHECK(cudaStreamWaitEvent(copy_stream, h->bcast_done[i], 0));
      auto e2 = UC::Trans::CudaSMCopyAsync(staging, (void**)list_ptrs,
                                           (size_t)block_bytes, (size_t)cnt, copy_stream);
      if (e2 != cudaSuccess) throw std::runtime_error("scatter failed");
      CUDA_CHECK(cudaEventRecord(h->scatter_done[i], copy_stream));
    }
  }

  // wait all work done (basic同步版)
  CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
  CUDA_CHECK(cudaStreamSynchronize(copy_stream));
}

py::bytes get_unique_id() {
  ncclUniqueId uid;
  NCCL_CHECK(ncclGetUniqueId(&uid));
  return py::bytes(reinterpret_cast<char*>(&uid), sizeof(uid));
}

// -------------------- pybind --------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_nccl_comm", &create_nccl_comm, "create nccl comm");
  m.def("destroy_nccl_comm", &destroy_nccl_comm, "destroy nccl comm");
  m.def("broadcast_discrete", &broadcast_discrete,
        "gather + ncclBroadcast + scatter (discrete pointers)");
  m.def("get_unique_id", &get_unique_id);
  m.def("broadcast_discrete_pipelined",
      &broadcast_discrete_pipelined,
      py::arg("handle"),
      py::arg("d_ptrs_u64"),
      py::arg("n_blocks"),
      py::arg("block_bytes"),
      py::arg("max_chunk_bytes"),
      py::arg("root"));
}
