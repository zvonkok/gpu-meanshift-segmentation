cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

include(CTest)

include_directories("/usr/include/cuda")
include_directories("/usr/share/cuda/samples/common/inc")

add_executable(ms meanshiftfilter.cpp meanshiftfilter_gold.cpp rgbluv.cpp edison.cpp rlist.cpp meanshiftfilter_kernel.cu )

