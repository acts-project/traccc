/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "opaque_queue.hpp"

#include "traccc/alpaka/utils/get_vecmem_resource.hpp"

namespace traccc::alpaka::details {

opaque_queue::opaque_queue(std::size_t device)
    : m_device{device}, m_queue(nullptr) {
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, device);
    m_queue = std::make_unique<Queue>(devAcc);

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    m_nativeQueue = static_cast<void*>(::alpaka::getNativeHandle(*(m_queue)));
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    auto nativeQueue = ::alpaka::getNativeHandle(*(m_queue));
    m_nativeQueue = reinterpret_cast<void*>(&nativeQueue);
#else
    m_nativeQueue = nullptr;
#endif
}

}  // namespace traccc::alpaka::details
