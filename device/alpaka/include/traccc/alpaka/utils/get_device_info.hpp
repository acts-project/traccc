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
#elif defined(ALPAKA_ACC_CPU_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)
using AccTag = ::alpaka::TagCpuSycl;
#elif defined(ALPAKA_ACC_GPU_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)
using AccTag = ::alpaka::TagGpuSyclIntel;
#elif defined(ALPAKA_ACC_FPGA_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_FPGA)
using AccTag = ::alpaka::TagFpgaSyclIntel;
#else
using AccTag = ::alpaka::TagCpuSerial;
#endif

/// Function that prints the current device information to the console.
/// Included as part of the traccc::alpaka namespace, to avoid having to include
/// alpaka headers in any users of the library.
void get_device_info();

}  // namespace traccc::alpaka
