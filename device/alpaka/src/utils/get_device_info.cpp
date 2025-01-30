/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "utils.hpp"

// Project include(s).
#include "traccc/alpaka/utils/get_device_info.hpp"

// System include(s).
#include <iostream>

namespace traccc::alpaka {

void get_device_info() {
    int device = 0;
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, 0u);
    std::cout << "Using Alpaka device: " << ::alpaka::getName(devAcc)
              << " [id: " << device << "] " << std::endl;
}

}  // namespace traccc::alpaka
