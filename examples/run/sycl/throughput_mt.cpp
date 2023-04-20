/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/throughput_mt_alt.hpp"
#include "full_chain_algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/sycl/host_memory_resource.hpp>

int main(int argc, char* argv[]) {

    // Execute the throughput test.
    static const bool use_host_caching = true;
    return traccc::throughput_mt_alt<traccc::sycl::full_chain_algorithm>(
        "Multi-threaded SYCL GPU throughput tests", argc, argv,
        use_host_caching);
}
