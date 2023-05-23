/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

namespace traccc::device {

template <typename config_t>
TRACCC_DEVICE inline void count_threads(
    std::size_t globalIndex, const config_t cfg,
    vecmem::data::vector_view<const unsigned int> n_measurements_view,
    const unsigned int& n_total_measurements,
    vecmem::data::vector_view<unsigned int> n_threads_view,
    unsigned int& n_measurements_per_thread, unsigned int& n_total_threads) {

    vecmem::device_vector<const unsigned int> n_measurements(
        n_measurements_view);
    vecmem::device_vector<unsigned int> n_threads(n_threads_view);

    const unsigned int n_params = n_measurements.size();

    if (globalIndex >= n_params) {
        return;
    }

    const unsigned int n_avg_meas_per_track = n_total_measurements / n_params;

    const unsigned int n_meas_per_thread =
        (n_avg_meas_per_track + cfg.n_avg_threads_per_track - 1) /
        cfg.n_avg_threads_per_track;

    if (globalIndex == 0) {
        n_measurements_per_thread = n_meas_per_thread;
    }

    n_threads.at(globalIndex) =
        (n_measurements.at(globalIndex) + n_meas_per_thread - 1) /
        n_meas_per_thread;

    vecmem::device_atomic_ref<unsigned int> num_total_threads(n_total_threads);
    num_total_threads.fetch_add(n_threads.at(globalIndex));
}

}  // namespace traccc::device
