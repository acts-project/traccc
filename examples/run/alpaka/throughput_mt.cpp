/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/throughput_mt.hpp"

#include "full_chain_algorithm.hpp"

// VecMem include(s).
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <vecmem/memory/hip/host_memory_resource.hpp>
#else
#include <vecmem/memory/host_memory_resource.hpp>
#endif

int main(int argc, char* argv[]) {

    // Execute the throughput test.
    static const bool use_host_caching = true;
    return traccc::throughput_mt<traccc::alpaka::full_chain_algorithm,
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                                 vecmem::cuda::host_memory_resource
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
                                 vecmem::hip::host_memory_resource
#else
                                 vecmem::host_memory_resource
#endif
                                 >("Multi-threaded Alpaka GPU throughput tests",
                                   argc, argv, use_host_caching);
}
