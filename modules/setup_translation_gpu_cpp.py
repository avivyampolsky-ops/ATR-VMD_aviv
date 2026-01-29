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

# Only add libraries if they are actually found by pkg-config or likely to exist
# to prevent linker errors if the user has a partial install.
# We trust the pkg-config output for what is available.
if "opencv_xfeatures2d" not in existing_libs and "opencv_xfeatures2d" in output:
     # Only force add it if we know it's there but maybe missed by our simple parsing
     # Actually, let's just trust existing_libs + what we verify.
     pass

# Force add specific libs only if we are reasonably sure they exist or if the user env is standard.
# For robustness, we will only add required libs that are NOT in existing_libs
# if we think they might be implicit.
# However, to respect "don't remove anything" but "let it work", we should
# try to filter out libs that definitely don't exist in the flags.

final_libs = []
for lib in opencv_flags["libraries"]:
    final_libs.append(lib)

# We iterate over our required list. If it's missing from the auto-detected list,
# we add it ONLY if it seems safe.
for lib in required_opencv_libs:
    if lib not in existing_libs:
        # Special check for xfeatures2d: if header was missing (likely), the lib might be missing too.
        # Adding it blindly causes linker error.
        # We can try to use ldconfig or just skip it if not found by pkg-config.
        if lib == "opencv_xfeatures2d":
             # Use a stricter check?
             # For now, if pkg-config didn't return it, we assume it's missing
             # and do NOT add it. The C++ code handles the missing header via __has_include.
             continue
        final_libs.append(lib)

opencv_flags["libraries"] = final_libs

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
