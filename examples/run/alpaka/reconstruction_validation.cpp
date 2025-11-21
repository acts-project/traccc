/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../common/device_reconstruction_validation.hpp"

// Local include(s).
#include "device_backend.hpp"

int main(int argc, char* argv[]) {

    return traccc::device_reconstruction_validation<
        traccc::alpaka::device_backend>("reconstruction_validation_alpaka",
                                        "Alpaka Reconstruction Validation",
                                        argc, argv);
}
