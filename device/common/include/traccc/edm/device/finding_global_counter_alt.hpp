/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

struct finding_global_counter_alt {

    // Total number of measurements associated with input parameters
    unsigned int n_total_measurements;

    // Number of parameters for the next step
    unsigned int n_out_params;
};

}  // namespace traccc::device
