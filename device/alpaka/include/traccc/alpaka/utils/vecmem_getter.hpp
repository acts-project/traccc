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
#include <sycl/sycl.hpp>
#include <vecmem/memory/sycl/device_memory_resource.hpp>
#include <vecmem/memory/sycl/host_memory_resource.hpp>
#include <vecmem/memory/sycl/shared_memory_resource.hpp>
#include <vecmem/utils/sycl/copy.hpp>
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
template <typename TTag>
std::shared_ptr<::vecmem::memory_resource> get_host_memory_resource() {
    return std::make_shared<::vecmem::host_memory_resource>();
}
template <typename TTag>
std::shared_ptr<::vecmem::memory_resource> get_device_memory_resource() {
    return std::make_shared<::vecmem::host_memory_resource>();
}

template <typename TTag>
std::shared_ptr<::vecmem::memory_resource> get_managed_memory_resource() {
    return std::make_shared<::vecmem::host_memory_resource>();
}

template <typename TTag>
std::shared_ptr<::vecmem::copy> get_device_copy() {
    return std::make_shared<::vecmem::copy>();
}

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
template <>
std::shared_ptr<::vecmem::memory_resource>
get_host_memory_resource<::alpaka::TagGpuCudaRt>() {
    return std::static_pointer_cast<::vecmem::memory_resource>(
        std::make_shared<::vecmem::cuda::host_memory_resource>());
}
template <>
std::shared_ptr<::vecmem::memory_resource>
get_device_memory_resource<::alpaka::TagGpuCudaRt>() {
    return std::static_pointer_cast<::vecmem::memory_resource>(
        std::make_shared<::vecmem::cuda::device_memory_resource>());
}

template <>
std::shared_ptr<::vecmem::memory_resource>
get_managed_memory_resource<::alpaka::TagGpuCudaRt>() {
    return std::static_pointer_cast<::vecmem::memory_resource>(
        std::make_shared<::vecmem::cuda::managed_memory_resource>());
}

template <>
std::shared_ptr<::vecmem::copy> get_device_copy<::alpaka::TagGpuCudaRt>() {
    return static_pointer_cast<::vecmem::copy>(
        std::make_shared<::vecmem::copy>(::vecmem::cuda::copy()));
}
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
template <>
std::shared_ptr<::vecmem::memory_resource>
get_host_memory_resource<::alpaka::TagGpuHipRt>() {
    return std::static_pointer_cast<::vecmem::memory_resource>(
        std::make_shared<::vecmem::hip::host_memory_resource>());
}
template <>
std::shared_ptr<::vecmem::memory_resource>
get_device_memory_resource<::alpaka::TagGpuHipRt>() {
    return std::static_pointer_cast<::vecmem::memory_resource>(
        std::make_shared<::vecmem::hip::device_memory_resource>());
}

template <>
std::shared_ptr<::vecmem::memory_resource>
get_managed_memory_resource<::alpaka::TagGpuHipRt>() {
    return std::static_pointer_cast<::vecmem::memory_resource>(
        std::make_shared<::vecmem::hip::managed_memory_resource>());
}

template <>
std::shared_ptr<::vecmem::copy> get_device_copy<::alpaka::TagGpuHipRt>() {
    return static_pointer_cast<::vecmem::copy>(
        std::make_shared<::vecmem::copy>(::vecmem::hip::copy()));
}

#elif defined(ALPAKA_ACC_GPU_SYCL_ENABLED)
template <>
std::shared_ptr<::vecmem::memory_resource>
get_host_memory_resource<::alpaka::TagGenericSycl>() {
    ::sycl::queue q;
    vecmem::sycl::queue_wrapper qw{&q};
    return std::static_pointer_cast<::vecmem::memory_resource>(
        std::make_shared<::vecmem::sycl::host_memory_resource>(qw));
}
template <>
std::shared_ptr<::vecmem::memory_resource>
get_device_memory_resource<::alpaka::TagGenericSycl>() {
    ::sycl::queue q;
    vecmem::sycl::queue_wrapper qw{&q};
    return std::static_pointer_cast<::vecmem::memory_resource>(
        std::make_shared<::vecmem::sycl::device_memory_resource>(qw));
}

template <>
std::shared_ptr<::vecmem::memory_resource>
get_managed_memory_resource<::alpaka::TagGenericSycl>() {
    ::sycl::queue q;
    vecmem::sycl::queue_wrapper qw{&q};
    return std::static_pointer_cast<::vecmem::memory_resource>(
        std::make_shared<::vecmem::sycl::managed_memory_resource>(qw));
}

template <>
std::shared_ptr<::vecmem::copy> get_device_copy<::alpaka::TagGenericSycl>() {
    ::sycl::queue q;
    vecmem::sycl::queue_wrapper qw{&q};
    return static_pointer_cast<::vecmem::copy>(
        std::make_shared<::vecmem::copy>(::vecmem::sycl::copy(qw)));
}
#endif

}  // namespace traccc::alpaka::vecmem
