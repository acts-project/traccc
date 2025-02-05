#pragma once

// VecMem include(s).
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <vecmem/memory/hip/device_memory_resource.hpp>
#include <vecmem/memory/hip/host_memory_resource.hpp>
#include <vecmem/memory/hip/managed_memory_resource.hpp>
#include <vecmem/utils/hip/copy.hpp>

#elif defined(ALPAKA_ACC_SYCL_ENABLED)
#include <vecmem/memory/sycl/device_memory_resource.hpp>
#include <vecmem/memory/sycl/host_memory_resource.hpp>
#include <vecmem/memory/sycl/shared_memory_resource.hpp>
#include <vecmem/utils/sycl/copy.hpp>
#include <sycl/sycl.hpp>
#include <vecmem/utils/sycl/queue_wrapper.hpp>

#else
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>
#endif

#include <alpaka/acc/Tag.hpp>
#include "traccc/alpaka/utils/device_tag.hpp"

// Forward declarations so we can compile the types below
namespace vecmem {
class host_memory_resource;
class copy;
namespace cuda {
class host_memory_resource;
class device_memory_resource;
class managed_memory_resource;
class copy;
}  // namespace cuda
namespace hip {
class host_memory_resource;
class device_memory_resource;
class managed_memory_resource;
class copy;
}  // namespace hip
namespace sycl {
class host_memory_resource;
class device_memory_resource;
class shared_memory_resource;
class copy;
}  // namespace sycl
}  // namespace vecmem

namespace traccc::alpaka::vecmem {
template<typename TTag>
void get_host_memory_resource(::vecmem::memory_resource& hmr)
{
    hmr = ::vecmem::host_memory_resource{};
}

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
template<>
void get_host_memory_resource<::alpaka::TagGpuCudaRt>(::vecmem::memory_resource& hmr)
{
    hmr = ::vecmem::cuda::host_memory_resource{};
}
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
template<>
void get_host_memory_resource<::alpaka::TagGpuHipRt>(::vecmem::memory_resource& hmr)
{
    hmr = ::vecmem::hip::host_memory_resource{};
}

#elif defined(ALPAKA_ACC_GPU_SYCL_ENABLED)
#if defined(ALPAKA_SYCL_ONEAPI_CPU)
template<>
void get_host_memory_resource<::alpaka::TagCpuSycl>(::vecmem::memory_resource& hmr)
{
    ::sycl::queue q;
    vecmem::sycl::queue_wrapper qw{&q};
    hmr = ::vecmem::sycl::host_memory_resource{qw};
}
#elif defined(ALPAKA_SYCL_ONEAPI_FPGA)
template<>
void get_host_memory_resource<::alpaka::TagFpgaSyclIntel>(::vecmem::memory_resource& hmr)
{
    ::sycl::queue q;
    vecmem::sycl::queue_wrapper qw{&q};
    hmr = ::vecmem::sycl::host_memory_resource{qw};
}
#elif defined(ALPAKA_SYCL_ONEAPI_GPU)
template<>
void get_host_memory_resource<::alpaka::TagGpuSyclIntel>(::vecmem::memory_resource& hmr)
{
    ::sycl::queue q;
    vecmem::sycl::queue_wrapper qw{&q};
    hmr = ::vecmem::sycl::host_memory_resource{qw};
}
#endif
#endif

}
