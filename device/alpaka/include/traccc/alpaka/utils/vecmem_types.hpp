/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

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
#include <vecmem/utils/sycl/copy.hpp>

#else
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>
#endif

#include <alpaka/alpaka.hpp>

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
class managed_memory_resource;
class copy;
}  // namespace sycl
}  // namespace vecmem

namespace traccc::alpaka::vecmem {
// For all CPU accelerators (except SYCL), just use host
template <typename T>
struct host_device_types {
    using device_memory_resource = ::vecmem::host_memory_resource;
    using host_memory_resource = ::vecmem::host_memory_resource;
    using managed_memory_resource = ::vecmem::host_memory_resource;
    using device_copy = ::vecmem::copy;
};
template <>
struct host_device_types<::alpaka::TagGpuCudaRt> {
    using device_memory_resource = ::vecmem::cuda::host_memory_resource;
    using host_memory_resource = ::vecmem::cuda::host_memory_resource;
    using managed_memory_resource = ::vecmem::cuda::managed_memory_resource;
    using device_copy = ::vecmem::cuda::copy;
};
template <>
struct host_device_types<::alpaka::TagGpuHipRt> {
    using device_memory_resource = ::vecmem::hip::device_memory_resource;
    using host_memory_resource = ::vecmem::hip::host_memory_resource;
    using managed_memory_resource = ::vecmem::hip::managed_memory_resource;
    using device_copy = ::vecmem::hip::copy;
};
template <>
struct host_device_types<::alpaka::TagCpuSycl> {
    using device_memory_resource = ::vecmem::sycl::device_memory_resource;
    using host_memory_resource = ::vecmem::sycl::host_memory_resource;
    using managed_memory_resource = ::vecmem::sycl::host_memory_resource;
    using device_copy = ::vecmem::sycl::copy;
};
template <>
struct host_device_types<::alpaka::TagFpgaSyclIntel> {
    using device_memory_resource = ::vecmem::sycl::device_memory_resource;
    using host_memory_resource = ::vecmem::sycl::host_memory_resource;
    using managed_memory_resource = ::vecmem::sycl::host_memory_resource;
    using device_copy = ::vecmem::sycl::copy;
};
template <>
struct host_device_types<::alpaka::TagGpuSyclIntel> {
    using device_memory_resource = ::vecmem::sycl::device_memory_resource;
    using host_memory_resource = ::vecmem::sycl::host_memory_resource;
    using device_copy = ::vecmem::sycl::copy;
};
}  // namespace traccc::alpaka::vecmem
