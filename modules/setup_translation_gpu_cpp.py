from __future__ import annotations

# Build in-place from this directory:
#   python setup_translation_gpu_cpp.py build_ext --inplace
# Requires: pybind11 and OpenCV (with CUDA headers/libs) available in the active env.

import subprocess
import sys
from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def _pkg_config_flags(package: str) -> dict:
    try:
        output = subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", package],
            text=True,
        ).strip()
    except Exception:
        return {
            "include_dirs": [],
            "library_dirs": [],
            "libraries": [],
            "extra_compile_args": [],
            "extra_link_args": [],
        }

    include_dirs = []
    library_dirs = []
    libraries = []
    extra_compile_args = []
    extra_link_args = []

    for flag in output.split():
        if flag.startswith("-I"):
            include_dirs.append(flag[2:])
        elif flag.startswith("-L"):
            library_dirs.append(flag[2:])
        elif flag.startswith("-l"):
            libraries.append(flag[2:])
        else:
            extra_compile_args.append(flag)
            extra_link_args.append(flag)

    return {
        "include_dirs": include_dirs,
        "library_dirs": library_dirs,
        "libraries": libraries,
        "extra_compile_args": extra_compile_args,
        "extra_link_args": extra_link_args,
    }


opencv_flags = _pkg_config_flags("opencv4")
if not opencv_flags["include_dirs"]:
    opencv_flags = _pkg_config_flags("opencv")
env_include = Path(sys.prefix) / "include" / "opencv4"
if env_include.exists():
    opencv_flags["include_dirs"].insert(0, str(env_include))
if "/usr/include/opencv4" not in opencv_flags["include_dirs"]:
    if Path("/usr/include/opencv4/opencv2/core.hpp").exists():
        opencv_flags["include_dirs"].append("/usr/include/opencv4")
env_lib = Path(sys.prefix) / "lib"
if env_lib.exists():
    opencv_flags["library_dirs"].insert(0, str(env_lib))
    opencv_flags["extra_link_args"].append(f"-Wl,-rpath,{env_lib}")

if env_include.exists():
    opencv_flags["include_dirs"] = [
        d for d in opencv_flags["include_dirs"]
        if d not in ("/usr/include/opencv4", "/usr/local/include/opencv4")
    ]

required_opencv_libs = [
    "opencv_core",
    "opencv_imgproc",
    "opencv_video",
    "opencv_calib3d",
    "opencv_features2d",
    "opencv_flann",
    "opencv_cudaarithm",
    "opencv_cudabgsegm",
    "opencv_cudafilters",
    "opencv_cudaimgproc",
    "opencv_cudawarping",
    "opencv_xfeatures2d",
]
existing_libs = set(opencv_flags["libraries"])
for lib in required_opencv_libs:
    if lib not in existing_libs:
        opencv_flags["libraries"].append(lib)

ext_modules = [
    Pybind11Extension(
        "_translation_gpu_cpp",
        [str(Path(__file__).parent / "translation_gpu_cpp.cpp")],
        include_dirs=opencv_flags["include_dirs"],
        library_dirs=opencv_flags["library_dirs"],
        libraries=opencv_flags["libraries"],
        extra_compile_args=opencv_flags["extra_compile_args"],
        extra_link_args=opencv_flags["extra_link_args"],
        cxx_std=17,
    ),
    Pybind11Extension(
        "_detector_cpp",
        [str(Path(__file__).parent / "detector_cpp.cpp")],
        include_dirs=opencv_flags["include_dirs"],
        library_dirs=opencv_flags["library_dirs"],
        libraries=opencv_flags["libraries"],
        extra_compile_args=opencv_flags["extra_compile_args"],
        extra_link_args=opencv_flags["extra_link_args"],
        cxx_std=17,
    ),
    Pybind11Extension(
        "_register_detect_cpp",
        [str(Path(__file__).parent / "register_detect_cpp.cpp")],
        include_dirs=opencv_flags["include_dirs"],
        library_dirs=opencv_flags["library_dirs"],
        libraries=opencv_flags["libraries"],
        extra_compile_args=opencv_flags["extra_compile_args"],
        extra_link_args=opencv_flags["extra_link_args"],
        cxx_std=17,
    ),
]

setup(
    name="translation_gpu_cpp",
    version="0.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
