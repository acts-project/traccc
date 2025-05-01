/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "utils.hpp"

// Project include(s).
#include "traccc/alpaka/utils/get_vecmem_resource.hpp"

// Standard library include(s)
#include <memory>
#include <stdexcept>

namespace traccc::alpaka::details {

// Per-accelerator types for vecmem.
template <typename T>
struct host_device_types {
    using device_memory_resource = vecmem::host_memory_resource;
    using host_memory_resource = vecmem::host_memory_resource;
    using managed_memory_resource = vecmem::host_memory_resource;
    using device_copy = vecmem::copy;
    using device_async_copy = vecmem::copy;
};
template <>
struct host_device_types<::alpaka::TagGpuCudaRt> {
    using device_memory_resource = vecmem::cuda::device_memory_resource;
    using host_memory_resource = vecmem::cuda::host_memory_resource;
    using managed_memory_resource = vecmem::cuda::managed_memory_resource;
    using device_copy = vecmem::cuda::copy;
    using device_async_copy = vecmem::cuda::async_copy;
};
template <>
struct host_device_types<::alpaka::TagGpuHipRt> {
    using device_memory_resource = vecmem::hip::device_memory_resource;
    using host_memory_resource = vecmem::hip::host_memory_resource;
    using managed_memory_resource = vecmem::hip::managed_memory_resource;
    using device_copy = vecmem::hip::copy;
    using device_async_copy = vecmem::hip::async_copy;
};
template <>
struct host_device_types<::alpaka::TagCpuSycl> {
    using device_memory_resource = vecmem::sycl::device_memory_resource;
    using host_memory_resource = vecmem::sycl::host_memory_resource;
    using managed_memory_resource = vecmem::sycl::shared_memory_resource;
    using device_copy = vecmem::sycl::copy;
    using device_async_copy = vecmem::sycl::async_copy;
};
template <>
struct host_device_types<::alpaka::TagFpgaSyclIntel> {
    using device_memory_resource = vecmem::sycl::device_memory_resource;
    using host_memory_resource = vecmem::sycl::host_memory_resource;
    using managed_memory_resource = vecmem::sycl::shared_memory_resource;
    using device_copy = vecmem::sycl::copy;
    using device_async_copy = vecmem::sycl::async_copy;
};
template <>
struct host_device_types<::alpaka::TagGpuSyclIntel> {
    using device_memory_resource = vecmem::sycl::device_memory_resource;
    using host_memory_resource = vecmem::sycl::host_memory_resource;
    using managed_memory_resource = vecmem::sycl::shared_memory_resource;
    using device_copy = vecmem::sycl::copy;
    using device_async_copy = vecmem::sycl::async_copy;
};

using AccTag = ::alpaka::AccToTag<Acc>;
using device_memory_resource =
    typename host_device_types<AccTag>::device_memory_resource;
using host_memory_resource =
    typename host_device_types<AccTag>::host_memory_resource;
using managed_memory_resource =
    typename host_device_types<AccTag>::managed_memory_resource;
using device_copy = typename host_device_types<AccTag>::device_copy;
using device_async_copy = typename host_device_types<AccTag>::device_async_copy;

struct vecmem_objects::impl {
    impl(traccc::alpaka::queue& queue) {
#ifdef ALPAKA_ACC_SYCL_ENABLED
        vecmem::sycl::queue_wrapper qw{queue.deviceNativeQueue()};

        m_host_mr = std::make_unique<host_memory_resource>(qw);
        m_device_mr = std::make_unique<device_memory_resource>(qw);
        m_managed_mr = std::make_unique<managed_memory_resource>(qw);
        m_copy = std::make_unique<device_copy>(qw);
        m_async_copy = std::make_unique<device_async_copy>(qw);
#else
        m_host_mr = std::make_unique<host_memory_resource>();
        m_device_mr = std::make_unique<device_memory_resource>();
        m_managed_mr = std::make_unique<managed_memory_resource>();
        m_copy = std::make_unique<device_copy>();
        if constexpr (std::is_same_v<device_copy, device_async_copy>) {
            m_async_copy = std::make_unique<device_copy>();
        } else {
            m_async_copy =
                std::make_unique<device_async_copy>(queue.deviceNativeQueue());
        }
#endif
    }

    ~impl() = default;

    std::unique_ptr<vecmem::memory_resource> m_host_mr;
    std::unique_ptr<vecmem::memory_resource> m_device_mr;
    std::unique_ptr<vecmem::memory_resource> m_managed_mr;
    std::unique_ptr<vecmem::copy> m_copy;
    std::unique_ptr<vecmem::copy> m_async_copy;
};

vecmem_objects::vecmem_objects(traccc::alpaka::queue& queue)
    : m_impl(std::make_unique<impl>(queue)) {}
vecmem_objects::~vecmem_objects() = default;

vecmem::memory_resource& vecmem_objects::host_mr() const {
    return *(m_impl->m_host_mr);
}

vecmem::memory_resource& vecmem_objects::device_mr() const {
    return *(m_impl->m_device_mr);
}

vecmem::memory_resource& vecmem_objects::managed_mr() const {
    return *(m_impl->m_managed_mr);
}

vecmem::copy& vecmem_objects::copy() const {
    return *(m_impl->m_copy);
}

vecmem::copy& vecmem_objects::async_copy() const {
    return *(m_impl->m_async_copy);
}

}  // namespace traccc::alpaka::details
