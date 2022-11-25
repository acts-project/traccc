/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/throughput_mt.hpp"

#include "full_chain_algorithm.hpp"

int main(int argc, char* argv[]) {

    // Execute the throughput test.
    return traccc::throughput_mt<traccc::full_chain_algorithm>(
        "Multi-threaded host-only throughput tests", argc, argv);
}
