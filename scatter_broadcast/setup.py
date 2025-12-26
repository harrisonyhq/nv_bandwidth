from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="discrete_nccl_bcast",
    ext_modules=[
        CUDAExtension(
            name="discrete_nccl_bcast",
            sources=[
                "discrete_bcast.cu",
                "cuda_sm_kernel.cu",   # ✅ 加上
            ],
            include_dirs=[
                ".",                   # ✅ 让 #include "cuda_sm_kernel.h" 找到
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
            },
            libraries=["nccl"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
