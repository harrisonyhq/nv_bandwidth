// exp_baseline.cu
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <numa.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CK_CUDA(x) do{ auto e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1);} }while(0)
#define CK_NCCL(x) do{ auto e=(x); if(e!=ncclSuccess){ \
  printf("NCCL error %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1);} }while(0)

int main(int argc, char** argv){
  MPI_Init(&argc,&argv);

  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&world);

  int mode = (argc > 1) ? atoi(argv[1]) : 0;
  double mib = (argc > 2) ? atof(argv[2]) : 512.0;
  int iters = (argc > 3) ? atoi(argv[3]) : 100;
  size_t bytes = (size_t) llround(mib * 1024.0 * 1024.0);

  if(numa_available() < 0){
    if(rank==0) printf("NUMA not available\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  CK_CUDA(cudaSetDevice(rank));

  // === NUMA node：简单固定为 0（baseline）===
  int numa_node = 1;
  numa_run_on_node(numa_node);

  // === Host pinned ===
  void* hptr = nullptr;
  if(mode == 1){
    // 同一物理页：/dev/shm 共享内存，所有进程 mmap 到各自地址空间
    const char* shm_name = "/bw_shm_pinbuf";
    int fd = -1;

    if(rank == 0){
      // 创建共享内存对象并扩容
      fd = shm_open(shm_name, O_CREAT | O_RDWR, 0600);
      if(fd < 0){ perror("shm_open"); MPI_Abort(MPI_COMM_WORLD, 1); }
      if(ftruncate(fd, (off_t)bytes) != 0){ perror("ftruncate"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    // 等待 rank0 创建完成
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank != 0){
      fd = shm_open(shm_name, O_RDWR, 0600);
      if(fd < 0){ perror("shm_open"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    hptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if(hptr == MAP_FAILED){ perror("mmap"); MPI_Abort(MPI_COMM_WORLD, 1); }
    close(fd);

    // first-touch：由 rank0 在绑定的 numa_node 上触碰整段内存，决定物理页 NUMA 落点
    if(rank == 0){
      numa_run_on_node(numa_node);
      memset(hptr, 1, bytes);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 每个进程都要 register 自己的 mapping
    CK_CUDA(cudaHostRegister(hptr, bytes, cudaHostRegisterPortable));
  } else {
    // 各自一块 pinned
    int cur_numa_node = 1;
    // if (rank < 4){
    //   cur_numa_node = 0;
    // }
    // else{
    //   cur_numa_node = 1;
    // }
    hptr = numa_alloc_onnode(bytes, cur_numa_node);
    memset(hptr, rank, bytes);
    CK_CUDA(cudaHostRegister(hptr, bytes, cudaHostRegisterPortable));
  }

  // === Device buffer ===
  void* dptr = nullptr;
  CK_CUDA(cudaMalloc(&dptr, bytes));

  cudaStream_t stream;
  CK_CUDA(cudaStreamCreate(&stream));
  cudaEvent_t st, ed;
  CK_CUDA(cudaEventCreate(&st));
  CK_CUDA(cudaEventCreate(&ed));

  // === baseline 0：单卡 ===
  if(mode == 0){
    if(rank != 0){
      MPI_Finalize();
      return 0;
    }
  }

  // === baseline 3：NCCL broadcast ===
  ncclComm_t comm;
  if(mode == 3){
    ncclUniqueId id;
    if(rank == 0) CK_NCCL(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    CK_NCCL(ncclCommInitRank(&comm, world, id, rank));
  }

  // warmup
  for(int i=0;i<10;i++){
    if(mode == 3){
      if(rank == 0)
        CK_CUDA(cudaMemcpyAsync(dptr, hptr, bytes, cudaMemcpyHostToDevice, stream));
      CK_NCCL(ncclBroadcast(dptr, dptr, bytes/4, ncclFloat,
                            0, comm, stream));
    } else {
      CK_CUDA(cudaMemcpyAsync(dptr, hptr, bytes, cudaMemcpyHostToDevice, stream));
    }
  }
  CK_CUDA(cudaStreamSynchronize(stream));

  CK_CUDA(cudaEventRecord(st, stream));
  for(int i=0;i<iters;i++){
    if(mode == 3){
      if(rank == 0)
        CK_CUDA(cudaMemcpyAsync(dptr, hptr, bytes, cudaMemcpyHostToDevice, stream));
      CK_NCCL(ncclBroadcast(dptr, dptr, bytes/4, ncclFloat,
                            0, comm, stream));
    } else {
      CK_CUDA(cudaMemcpyAsync(dptr, hptr, bytes, cudaMemcpyHostToDevice, stream));
    }
  }
  CK_CUDA(cudaEventRecord(ed, stream));
  CK_CUDA(cudaEventSynchronize(ed));

  float ms;
  CK_CUDA(cudaEventElapsedTime(&ms, st, ed));

  double sec = ms / 1e3;
  double gb = (double)bytes * iters / 1e9;

  printf("[rank %d] mode=%d BW = %.2f GB/s\n", rank, mode, gb/sec);

  if(mode == 3) ncclCommDestroy(comm);

  MPI_Finalize();
  return 0;
}
