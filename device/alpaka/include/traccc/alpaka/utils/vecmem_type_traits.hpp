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

#include <type_traits>

namespace traccc::alpaka::vecmem {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    struct host_device_traits {
      using device_memory_resource = ::vecmem::cuda::device_memory_resource;
      using host_memory_resource = ::vecmem::cuda::host_memory_resource ;
      using managed_memory_resource = ::vecmem::cuda::managed_memory_resource;
      using device_copy = ::vecmem::cuda::copy;
    };  // struct host_device_traits
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    struct host_device_traits {
      using device_memory_resource = ::vecmem::hip::device_memory_resource;
      using host_memory_resource = ::vecmem::hip::host_memory_resource ;
      using managed_memory_resource = ::vecmem::hip::managed_memory_resource;
      using device_copy = ::vecmem::hip::copy;
    };  // struct host_device_traits
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    struct host_device_traits {
      using device_memory_resource = ::vecmem::sycl::device_memory_resource;
      using host_memory_resource = ::vecmem::sycl::host_memory_resource ;
      using managed_memory_resource = ::vecmem::sycl::managed_memory_resource;
      using device_copy = ::vecmem::sycl::copy;
    };  // struct host_device_traits
#else  // host-only
    struct host_device_traits {
      using device_memory_resource = ::vecmem::host_memory_resource;
      using host_memory_resource = ::vecmem::host_memory_resource ;
      using managed_memory_resource = ::vecmem::managed_memory_resource;
      using device_copy = ::vecmem::copy;
    };  // struct host_device_traits
#endif
}  // namespace traccc::alpaka::vecmem
