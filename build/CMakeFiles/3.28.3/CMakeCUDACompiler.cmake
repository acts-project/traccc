set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/usr/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "12.6.77")
set(CMAKE_CUDA_DEVICE_LINKER "/usr/local/cuda-12.6/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/usr/local/cuda-12.6/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17;cuda_std_20")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "cuda_std_20")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "13.2")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)
set(CMAKE_CUDA_LINKER_DEPFILE_SUPPORTED )

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/usr/local/cuda-12.6")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/usr/local/cuda-12.6")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "12.6.77")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/usr/local/cuda-12.6")

set(CMAKE_CUDA_ARCHITECTURES_ALL "50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87-real;89-real;90")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "50-real;60-real;70-real;80-real;90")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "89-real")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/local/cuda-12.6/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs;/usr/local/cuda-12.6/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/usr/include/c++/13;/usr/include/x86_64-linux-gnu/c++/13;/usr/include/c++/13/backward;/usr/lib/gcc/x86_64-linux-gnu/13/include;/usr/local/include;/usr/include/x86_64-linux-gnu;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs;/usr/local/cuda-12.6/targets/x86_64-linux/lib;/opt/intel/oneapi/tbb/2021.13/lib/intel64/lib;/opt/intel/oneapi/mpi/2021.13/lib;/opt/intel/oneapi/mkl/2024.2/lib;/opt/intel/oneapi/ippcp/2021.12/lib;/opt/intel/oneapi/ipp/2021.12/lib;/opt/intel/oneapi/dpl/2022.6/lib;/opt/intel/oneapi/dnnl/2024.2/lib;/opt/intel/oneapi/dal/2024.6/lib;/opt/intel/oneapi/compiler/2024.2/lib;/opt/intel/oneapi/ccl/2021.13/lib;/usr/lib/gcc/x86_64-linux-gnu/13;/usr/lib/x86_64-linux-gnu;/usr/lib;/lib/x86_64-linux-gnu;/lib;/opt/intel/oneapi/tbb/2021.13/lib/intel64/gcc4.8;/opt/intel/oneapi/ippcp/2021.12/lib32")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/local/bin/llvm-ar")
set(CMAKE_MT "")
