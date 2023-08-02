/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/throughput_st.hpp"

#include "full_chain_algorithm.hpp"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#else
#include <vecmem/memory/host_memory_resource.hpp>
#endif

int main(int argc, char* argv[]) {

    // Execute the throughput test.
    static const bool use_host_caching = true;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    return traccc::throughput_st<
        traccc::alpaka::full_chain_algorithm,
        vecmem::cuda::host_memory_resource
    >(
        "Single-threaded CUDA GPU throughput tests", argc, argv, use_host_caching
    );
#else
    return traccc::throughput_st<
        traccc::alpaka::full_chain_algorithm,
        vecmem::cuda::host_memory_resource
        vecmem::host_memory_resource
    >(
        "Single-threaded CUDA GPU throughput tests", argc, argv, use_host_caching
    );
#endif
}
