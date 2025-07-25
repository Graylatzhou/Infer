from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

# 自动查找所有源文件和头文件目录
root_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(root_dir, "src")
include_dir = os.path.join(root_dir, "include")
cutlass_include_dir = os.path.join(root_dir, "third_party", "cutlass", "include") #定义 cutlass 目录

source_files = glob.glob(os.path.join(src_dir, '**', '*.cpp'), recursive=True) + \
               glob.glob(os.path.join(src_dir, '**', '*.cu'), recursive=True)

print("Found source files:", source_files)
print("Include directory:", include_dir)

setup(
    name='infer_ops',
    ext_modules=[
        CUDAExtension(
            name='infer_ops._C',
            sources=source_files,
            include_dirs=[include_dir, cutlass_include_dir],
            extra_compile_args={
                'cxx': [
                    '-std=c++17',
                    '-O3',
                    '-Wall', # 开启所有警告，帮助发现潜在问题
                ],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-g',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    }
)
