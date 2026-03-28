# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

_ROOT = Path(__file__).resolve().parent
_ext_src_root = _ROOT / "_ext_src"
_ext_include_dir = _ext_src_root / "include"
_ext_sources = glob.glob(str(_ext_src_root / "src" / "*.cpp")) + glob.glob(
    str(_ext_src_root / "src" / "*.cu")
)
_ext_headers = glob.glob(str(_ext_include_dir / "*"))

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            include_dirs=[str(_ext_include_dir)],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
