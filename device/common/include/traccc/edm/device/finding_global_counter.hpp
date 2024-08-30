/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

struct finding_global_counter {

    // Number of found measurements for the current step
    unsigned int n_candidates;

    // Number of parameters for the next step
    unsigned int n_out_params;

    // Number of valid tracks that meet criteria
    unsigned int n_valid_tracks;
};

}  // namespace traccc::device
