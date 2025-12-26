# nv_bandwidth
prerequests:

```bash
apt-get install libnccl2 libnccl-dev
```

compile:

```bash
nvcc -O3 -std=c++17 -c bw_test_multi.cu -o bw_test.o   -I/usr/local/openmpi/include
/usr/local/openmpi/bin/mpicxx -O3 bw_test.o -o exp_baseline   -L/usr/local/cuda-12.2/targets/x86_64-linux/lib -lcudart   -lnuma -lnccl
```

run:

```bash
# args: mode min_mib max_mib iters
CUDA_VISIBLE_DEVICES=1  mpirun -n 1 --allow-run-as-root ./exp_baseline 0 0.25 512 100
```

mode:
| mode | test case |
| ---- | -------------------------------------- |
 | 0 | Single GPU load pinned memory| 
 | 1 | Multi GPU load **same** share memory| 
 | 2 | Multi GPU load **different** pinned memory|
| 3 | Pinned → GPU0 → NCCL broadcast to other GPUs |