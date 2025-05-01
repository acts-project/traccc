/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/alpaka/utils/queue.hpp"

// VecMem include(s).
#if defined(TRACCC_BUILD_CUDA)
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>
#endif

#if defined(TRACCC_BUILD_HIP)
#include <vecmem/memory/hip/device_memory_resource.hpp>
#include <vecmem/memory/hip/host_memory_resource.hpp>
#include <vecmem/memory/hip/managed_memory_resource.hpp>
#include <vecmem/utils/hip/async_copy.hpp>
#include <vecmem/utils/hip/copy.hpp>
#endif

#if defined(TRACCC_BUILD_SYCL)
#include <vecmem/memory/sycl/device_memory_resource.hpp>
#include <vecmem/memory/sycl/host_memory_resource.hpp>
#include <vecmem/memory/sycl/shared_memory_resource.hpp>
#include <vecmem/utils/sycl/async_copy.hpp>
#include <vecmem/utils/sycl/copy.hpp>
#endif

#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// Standard library includes
#include <memory>

// Forward declarations so we can compile the types below
namespace vecmem {
class host_memory_resource;
class copy;
namespace cuda {
class host_memory_resource;
class device_memory_resource;
class managed_memory_resource;
class copy;
class async_copy;
}  // namespace cuda
namespace hip {
class host_memory_resource;
class device_memory_resource;
class managed_memory_resource;
class copy;
class async_copy;
}  // namespace hip
namespace sycl {
class host_memory_resource;
class device_memory_resource;
class shared_memory_resource;
class copy;
class async_copy;
}  // namespace sycl
}  // namespace vecmem

namespace traccc::alpaka::details {

/**
 * @brief Class that creates and owns vecmem resources, providing a generic
 * interface, such that a traccc::alpaka user does not need to know about the
 * underlying implementation.
 */
class vecmem_objects {

    public:
    vecmem_objects(traccc::alpaka::queue& queue);
    ~vecmem_objects();

    vecmem_objects(const vecmem_objects&) = delete;
    vecmem_objects& operator=(const vecmem_objects&) = delete;
    vecmem_objects(vecmem_objects&&) = delete;
    vecmem_objects& operator=(vecmem_objects&&) = delete;

    vecmem::memory_resource& host_mr() const;
    vecmem::memory_resource& device_mr() const;
    vecmem::memory_resource& managed_mr() const;
    vecmem::copy& copy() const;
    vecmem::copy& async_copy() const;

    private:
    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace traccc::alpaka::details
