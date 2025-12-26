// exp_baseline_sweep.cu
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <numa.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CK_CUDA(x) do{ auto e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  MPI_Abort(MPI_COMM_WORLD,1);} }while(0)

#define CK_NCCL(x) do{ auto e=(x); if(e!=ncclSuccess){ \
  printf("NCCL error %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(e)); \
  MPI_Abort(MPI_COMM_WORLD,1);} }while(0)

static inline size_t align_up(size_t x, size_t a) {
  return (x + a - 1) / a * a;
}

int main(int argc, char** argv){
  MPI_Init(&argc,&argv);

  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&world);

  // argv:
  //   mode (required-ish)
  //   min_mib (optional, default 0.25)
  //   max_mib (optional, default 512)
  //   iters   (optional, default 100)
  int mode        = (argc > 1) ? atoi(argv[1]) : 0;
  double min_mib  = (argc > 2) ? atof(argv[2]) : 0.25;
  double max_mib  = (argc > 3) ? atof(argv[3]) : 512.0;
  int iters       = (argc > 4) ? atoi(argv[4]) : 100;

  if (min_mib <= 0.0 || max_mib < min_mib) {
    if (rank == 0) fprintf(stderr, "Invalid mib range: min=%f max=%f\n", min_mib, max_mib);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if(numa_available() < 0){
    if(rank==0) printf("NUMA not available\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  int device_count = 0;
  CK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    if (rank == 0) fprintf(stderr, "No CUDA devices\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int dev = rank % device_count;
  CK_CUDA(cudaSetDevice(dev));

  int numa_node = 0;
  // if (rank < 4) cur_numa_node = 0; else cur_numa_node = 1;
  if(mode == 0 && rank != 0){
    MPI_Finalize();
    return 0;
  }

  ncclComm_t comm{};
  if(mode == 3){
    ncclUniqueId id;
    if(rank == 0) CK_NCCL(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    CK_NCCL(ncclCommInitRank(&comm, world, id, rank));
  }

  cudaStream_t stream;
  CK_CUDA(cudaStreamCreate(&stream));
  cudaEvent_t st, ed;
  CK_CUDA(cudaEventCreate(&st));
  CK_CUDA(cudaEventCreate(&ed));

  if (rank == 0) {
    printf("mode,rank,world,dev,numa_node,mib,bytes_aligned,BW_GBps\n");
    fflush(stdout);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  for(double mib = min_mib; mib <= max_mib + 1e-12; mib *= 2.0){
    size_t bytes = (size_t) llround(mib * 1024.0 * 1024.0);
    if (bytes == 0) bytes = 1;

    size_t bytes_aligned = align_up(bytes, 4);
    size_t count_f32 = bytes_aligned / 4;

    // --- Host buffer ---
    void* hptr = nullptr;
    std::string shm_name;

    if(mode == 1){
      shm_name = "/bw_shm_pinbuf_" + std::to_string((unsigned long long)bytes_aligned);
      const char* name = shm_name.c_str();

      int fd = -1;
      if(rank == 0){
        shm_unlink(name);

        fd = shm_open(name, O_CREAT | O_RDWR, 0600);
        if(fd < 0){ perror("shm_open"); MPI_Abort(MPI_COMM_WORLD, 1); }
        if(ftruncate(fd, (off_t)bytes_aligned) != 0){ perror("ftruncate"); MPI_Abort(MPI_COMM_WORLD, 1); }
      }

      MPI_Barrier(MPI_COMM_WORLD);

      if(rank != 0){
        fd = shm_open(name, O_RDWR, 0600);
        if(fd < 0){ perror("shm_open"); MPI_Abort(MPI_COMM_WORLD, 1); }
      }

      hptr = mmap(nullptr, bytes_aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if(hptr == MAP_FAILED){ perror("mmap"); MPI_Abort(MPI_COMM_WORLD, 1); }
      close(fd);

      if(rank == 0){
        if (numa_run_on_node(numa_node) != 0) {
          perror("numa_run_on_node");
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
        memset(hptr, 1, bytes_aligned);
      }
      MPI_Barrier(MPI_COMM_WORLD);

      CK_CUDA(cudaHostRegister(hptr, bytes_aligned, cudaHostRegisterPortable));
    } else {
      hptr = numa_alloc_onnode(bytes_aligned, numa_node);
      if (!hptr) {
        if (rank == 0) fprintf(stderr, "numa_alloc_onnode failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      memset(hptr, rank & 0xFF, bytes_aligned);
      CK_CUDA(cudaHostRegister(hptr, bytes_aligned, cudaHostRegisterPortable));
    }

    // --- Device buffer ---
    void* dptr = nullptr;
    CK_CUDA(cudaMalloc(&dptr, bytes_aligned));

    // warmup
    for(int i=0;i<10;i++){
      if(mode == 3){
        if(rank == 0)
          CK_CUDA(cudaMemcpyAsync(dptr, hptr, bytes_aligned, cudaMemcpyHostToDevice, stream));
        CK_NCCL(ncclBroadcast(dptr, dptr, count_f32, ncclFloat, 0, comm, stream));
      } else {
        CK_CUDA(cudaMemcpyAsync(dptr, hptr, bytes_aligned, cudaMemcpyHostToDevice, stream));
      }
    }
    CK_CUDA(cudaStreamSynchronize(stream));

    CK_CUDA(cudaEventRecord(st, stream));
    for(int i=0;i<iters;i++){
      if(mode == 3){
        if(rank == 0)
          CK_CUDA(cudaMemcpyAsync(dptr, hptr, bytes_aligned, cudaMemcpyHostToDevice, stream));
        CK_NCCL(ncclBroadcast(dptr, dptr, count_f32, ncclFloat, 0, comm, stream));
      } else {
        CK_CUDA(cudaMemcpyAsync(dptr, hptr, bytes_aligned, cudaMemcpyHostToDevice, stream));
      }
    }
    CK_CUDA(cudaEventRecord(ed, stream));
    CK_CUDA(cudaEventSynchronize(ed));

    float ms = 0.0f;
    CK_CUDA(cudaEventElapsedTime(&ms, st, ed));
    double sec = ms / 1e3;
    double gb  = (double)bytes_aligned * iters / 1e9;
    double bw  = gb / sec;

    printf("%d,%d,%d,%d,%d,%.6f,%llu,%.3f\n",
           mode, rank, world, dev, numa_node,
           mib,
           (unsigned long long)bytes_aligned,
           bw);
    fflush(stdout);

    // --- cleanup per size ---
    CK_CUDA(cudaFree(dptr));
    CK_CUDA(cudaHostUnregister(hptr));

    if(mode == 1){
      munmap(hptr, bytes_aligned);
      MPI_Barrier(MPI_COMM_WORLD);
      if(rank == 0){
        shm_unlink(shm_name.c_str());
      }
      MPI_Barrier(MPI_COMM_WORLD);
    } else {
      numa_free(hptr, bytes_aligned);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  if(mode == 3) ncclCommDestroy(comm);

  CK_CUDA(cudaEventDestroy(st));
  CK_CUDA(cudaEventDestroy(ed));
  CK_CUDA(cudaStreamDestroy(stream));

  MPI_Finalize();
  return 0;
}
