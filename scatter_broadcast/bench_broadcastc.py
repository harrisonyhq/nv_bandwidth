#!/usr/bin/env python3
# benchmark_broadcast.py
import argparse
import os
import time
import torch
import torch.distributed as dist
import numpy as np
from typing import List
import discrete_nccl_bcast as ext

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch broadcast bandwidth benchmark")
    parser.add_argument("--size-mb", type=float, required=True,
                        help="tensor size in MB (float32)")
    parser.add_argument("--backend", type=str, default="nccl",
                        choices=["nccl", "hccl", "gloo", "mpi"],
                        help="distributed backend")
    parser.add_argument("--warmup", type=int, default=5,
                        help="warmup iterations")
    parser.add_argument("--iters", type=int, default=20,
                        help="benchmark iterations")
    parser.add_argument("--tp-size", type=int, default=None,
                        help="TP group size (default: use all ranks)")
    return parser.parse_args()

def custom_broadcast(
    handle,
    d_ptrs_u64,
    n_blocks,
    block_bytes,
    root,
):
    # return dist._broadcast_coalesced(
    #     process_group=group,
    #     tensors=tensor_list,
    #     buffer_size=buffer_size,
    #     src=src
    # )
    ext.broadcast_discrete(
        handle,
        d_ptrs_u64,          # CUDA uint64 tensor
        n_blocks,
        block_bytes,
        root,
        0,
    )

def custom_broadcast1(
    handle,
    d_ptrs_u64,          # CUDA uint64 tensor
    n_blocks,
    block_bytes,
    max_chunk_bytes,    # max_chunk_bytes
    root,                   # root
):
    ext.broadcast_discrete_pipelined(
        handle,
        d_ptrs_u64,          # CUDA uint64 tensor
        n_blocks,
        block_bytes,
        max_chunk_bytes,    # max_chunk_bytes
        root,                   # root
    )
    



def _ensure_buffer(buffer, buffer_size, total_numel: int, device):
    """
    Initialize or ensure buffer for broadcast;
    Typically this buffer length equals to one layer kv cache tensor size.
    """
    if buffer is None or buffer_size < total_numel:
        buffer = torch.empty(
            total_numel,
            dtype=torch.bfloat16,
            device=device,
        )
        buffer_size = total_numel
    return buffer, buffer_size

def _broadcast(dst_tensor_addr: List[torch.Tensor], group, rank, buffer):
    """
    Broadcast tensor list in tp group.
    """
    rec_tensor = None
    total_numel = len(dst_tensor_addr) * dst_tensor_addr[0].numel()
    if rank == 0:
        tensor_to_broadcast = torch.stack(dst_tensor_addr)
        handle = torch.distributed.broadcast(
            tensor_to_broadcast, src=0, async_op=True, group=group
        )
    else:
        shape = (len(dst_tensor_addr),) + dst_tensor_addr[0].shape
        rec_tensor = buffer[:total_numel].view(shape)
        handle = torch.distributed.broadcast(
            rec_tensor, src=0, async_op=True, group=group
        )
    return handle, rec_tensor

def _broadcast_layers(dst_tensor_addr: list[torch.Tensor], group, rank, buffer):
    """
    Broadcast kv caches by layer.
    """
    num_layers = 27
    total = len(dst_tensor_addr)
    assert num_layers > 0 and total % num_layers == 0, (num_layers, total)
    num_tensors_per_layer = total // num_layers

    for layer_i in range(num_layers):
        start = layer_i * num_tensors_per_layer
        handle, rec_tensor = _broadcast(
            dst_tensor_addr[start : start + num_tensors_per_layer],
            group,
            rank,
            buffer
        )
        handle.wait()
        if rank != 0 and rec_tensor is not None:
            rec_tensor_list = list(torch.unbind(rec_tensor, dim=0))
            torch._foreach_copy_(
                dst_tensor_addr[start : start + num_tensors_per_layer],
                rec_tensor_list,
            )

def _broadcast_req(dst_tensor_addr: list[torch.Tensor], group, rank, buffer):
    # torch.distributed._broadcast_coalesced(self.group_coordinator.device_group,dst_tensor_addr, 5485760, 0)
    handle, rec_tensor = _broadcast(
            dst_tensor_addr,
            group,
            rank,
            buffer
        )
    handle.wait()
    if rank != 0 and rec_tensor is not None:
        rec_tensor_list = list(torch.unbind(rec_tensor, dim=0))
        torch._foreach_copy_(
            dst_tensor_addr,
            rec_tensor_list,
        )

def main():
    args = get_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch_dev = torch.npu if args.backend == "hccl" else torch.cuda
    platform = "npu" if args.backend == "hccl" else "cuda"
    torch_dev.set_device(local_rank)
    device=torch_dev.current_device()

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    buffer = None
    buffer_size = 0
    
    # 构建TP分组（核心改动：支持torchrun的TP分组）
    tp_size = args.tp_size if args.tp_size else world_size
    if world_size % tp_size != 0:
        raise ValueError(f"World size {world_size} must be divisible by TP size {tp_size}")
    # 按rank划分TP组（示例：连续rank为一组）
    tp_group_id = rank // tp_size
    tp_group_ranks = list(range(tp_group_id * tp_size, (tp_group_id + 1) * tp_size))
    tp_group = dist.new_group(ranks=tp_group_ranks) if rank in tp_group_ranks else None
    # TP组内局部rank（src用局部rank 0）
    tp_rank = rank - tp_group_id * tp_size if tp_group else -1
    if rank == 0:
        uid = ext.get_unique_id()  # bytes, len == sizeof(ncclUniqueId)
    else:
        uid = b""  # placeholder
    uid_list = [uid]
    dist.broadcast_object_list(uid_list, src=0)
    uid = uid_list[0]
    handle = ext.create_nccl_comm(world_size, rank, uid)
    # 构造tensor list（核心改动：从单tensor改为list）
    elem_size = 2  # bfloat16
    layer = 27
    blocks = 62
    dim = 576
    blk_size = 64
    num_tensors = layer * blocks
    numel_per_tensor = dim * blk_size
    if tp_rank == 0:  # TP组内src=0初始化
        tensor_list = [torch.zeros(numel_per_tensor, dtype=torch.bfloat16, device=f"{platform}:{device}") for _ in range(num_tensors)]
    else:
        tensor_list = [torch.ones(numel_per_tensor, dtype=torch.bfloat16, device=f"{platform}:{device}") for _ in range(num_tensors)]
    
    # 总量校验（确保和原脚本一致）
    total_numel = sum(t.numel() for t in tensor_list)
    buffer, buffer_size = _ensure_buffer(buffer, buffer_size, total_numel, device)
    print(f"before rank {rank} (tp_rank {tp_rank}) tensor sum: {tensor_list[0]} (total size: {total_numel*2/1024/1024}MB)")
    # buffer,buffer_size =  _ensure_buffer(buffer, buffer_size, numel, device)
    ptrs = np.array([t.data_ptr() for t in tensor_list], dtype=np.uint64)
    d_ptrs_u64 = torch.from_numpy(ptrs).to("cuda", dtype=torch.uint64)
    lens = np.array([t.numel() for t in tensor_list], dtype=np.int64)
    # warmup（改为tensor list + TP组）
    if tp_group:  # 仅TP组内进程执行
        for _ in range(args.warmup):
            # _broadcast_layers(tensor_list, tp_group, rank, buffer)
            # custom_broadcast1(handle,d_ptrs_u64,d_ptrs_u64.numel(),numel_per_tensor*2, 64 * 1024 * 1024, 0)
            # custom_broadcast(handle,d_ptrs_u64,d_ptrs_u64.numel(),numel_per_tensor*2, 0)
            _broadcast_req(tensor_list, tp_group, rank, buffer)

    torch_dev.synchronize()
    # benchmark（核心逻辑不变，仅传tensor list和TP组）
    start = time.perf_counter()
    if tp_group:
        for _ in range(args.iters):
            # _broadcast_layers(tensor_list, tp_group, rank, buffer)
            # custom_broadcast(handle,ptrs,lens)
            # custom_broadcast1(handle,d_ptrs_u64,d_ptrs_u64.numel(),numel_per_tensor*2, 64 * 1024 * 1024, 0)
            # custom_broadcast(handle,d_ptrs_u64,d_ptrs_u64.numel(),numel_per_tensor*2, 0)
            _broadcast_req(tensor_list, tp_group, rank, buffer)
    torch_dev.synchronize()
    elapsed = time.perf_counter() - start

    # 计算带宽（总量不变，逻辑复用）
    bytes_per_iter = total_numel * elem_size
    total_bytes = bytes_per_iter * args.iters
    bw_gbps = total_bytes / elapsed / 1e9 if elapsed > 0 else 0

    # if rank == 0:
    print(f"rank {rank} [summary] backend={args.backend}  size={args.size_mb:.1f}MB  "
            f"tp_size={tp_size}  world_size={world_size}  iters={args.iters}")
    print(f"rank {rank}  avg_time={elapsed/args.iters*1e3:.3f}ms  "
            f"bandwidth={bw_gbps:.2f} GB/s")
    print(f"after rank {rank} (tp_rank {tp_rank}) tensor sum: {tensor_list[0]}")

    if tp_group:
        dist.destroy_process_group(tp_group)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()