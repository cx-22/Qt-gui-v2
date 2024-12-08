from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "cx22",
        ["cx22.cpp"],
        include_dirs=["/home/eliastc2/.local/lib/python3.10/site-packages/pybind11/include", "/usr/include/python3.10", "/usr/include/opencv4"],
        library_dirs=["/usr/lib/x86_64-linux-gnu"],
        libraries=["opencv_core", "opencv_imgproc", "opencv_highgui"],
        extra_compile_args=["-std=c++17", "-Wall", "-v"],  # Add verbose output
        extra_link_args=["-v"],  # Add verbose output for linking as well
    ),
]

setup(
    name="cx22",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext}
)