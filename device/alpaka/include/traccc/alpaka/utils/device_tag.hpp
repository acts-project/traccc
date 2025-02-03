/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <alpaka/acc/Tag.hpp>

namespace traccc::alpaka {

// Get alpaka tag for current device
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
using AccTag = ::alpaka::TagGpuCudaRt;
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
using AccTag = ::alpaka::TagGpuHipRt;
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
#if defined(ALPAKA_SYCL_ONEAPI_CPU)
using AccTag = ::alpaka::TagCpuSycl;
#elif defined(ALPAKA_SYCL_ONEAPI_FPGA)
using AccTag = ::alpaka::TagFpgaSyclIntel;
#elif defined(ALPAKA_SYCL_ONEAPI_GPU)
using AccTag = ::alpaka::TagGpuSyclIntel;
#endif
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
using AccTag = ::alpaka::TagCpuThreads;
#endif

}  // namespace traccc::alpaka
