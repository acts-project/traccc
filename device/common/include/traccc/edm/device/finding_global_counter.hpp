/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

struct finding_global_counter {

    // Total number of measurements associated with input parameters
    unsigned int n_total_measurements;

    // Number of measurements per thread
    unsigned int n_measurements_per_thread;

    // Number of threads for find_track kernel
    unsigned int n_total_threads;

    // Divider for the number of measurements
    unsigned int divider;

    // Number of found measurements for the current step
    unsigned int n_candidates;

    // Number of parameters for the next step
    unsigned int n_out_params;
};

}  // namespace traccc::device
