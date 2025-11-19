/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../common/device_track_finding_validation.hpp"

// Local include(s).
#include "algorithm_maker.hpp"

int main(int argc, char* argv[]) {

    return traccc::device_track_finding_validation<
        traccc::cuda::algorithm_maker>("track_finding_validation_cuda",
                                       "CUDA Track Finding Validation", argc,
                                       argv);
}
