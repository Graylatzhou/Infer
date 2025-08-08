from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os
import torch

# 自动查找所有源文件和头文件目录
root_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(root_dir, "src")
include_dir = os.path.join(root_dir, "include")
cutlass_include_dir = os.path.join(root_dir, "third_party", "cutlass", "include") #定义 cutlass 目录

source_files = glob.glob(os.path.join(src_dir, '**', '*.cpp'), recursive=True) + \
               glob.glob(os.path.join(src_dir, '**', '*.cu'), recursive=True)
            #    glob.glob(os.path.join(src_dir, 'operators', 'flashattn', 'cutlass_impl', '*.cu'), recursive=True) + \
            #    glob.glob(os.path.join(src_dir, 'operators', 'flashattn', 'cutlass_impl', '*.cpp'), recursive=True)

def get_cuda_arch():
    try:
        arch_list = []
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            arch_list.append(f'{capability[0]}{capability[1]}')
        return arch_list if arch_list else ['80', '90']  # 默认支持 A100, H100
    except:
        return ['80', '90']

cuda_archs = ['80', '90']
gencode_flags = [f'-gencode=arch=compute_{arch},code=sm_{arch}' for arch in cuda_archs]

print("Found source files:", source_files)
print("Include directory:", include_dir)

setup(
    name='infer_ops',
    ext_modules=[
        CUDAExtension(
            name='infer_ops._C',
            sources=source_files,
            include_dirs=[
                include_dir, 
                cutlass_include_dir,
                # 确保 PyTorch 头文件在前面
                *torch.utils.cpp_extension.include_paths(),
            ],
            extra_compile_args={
                'cxx': [
                    '-std=c++17',
                    '-O3',
                    '-fPIC',
                    '-Wall',
                    '-Wno-unused-function',
                    '-Wno-unused-variable',
                    # 添加预处理器定义
                    '-DCUTE_NAMESPACE=cute',
                    '-DTORCH_EXTENSION_NAME=infer_ops',
                ],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '--use_fast_math',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    # 添加架构支持
                    *gencode_flags,
                    # 添加预处理器定义
                    '-DCUTE_NAMESPACE=cute',
                    '-DTORCH_EXTENSION_NAME=infer_ops',
                    # 增加共享内存限制
                    '-maxrregcount=255',
                    # 调试信息（可选）
                    '-lineinfo',
                ]
            },
            # 添加链接器选项
            extra_link_args=['-lcudart', '-lcublas', '-lcurand'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True, max_jobs=4)
    }
)

