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
    bound_track_parameters_collection_types::const_view params_view,
    vecmem::data::vector_view<const detray::geometry::barcode> barcodes_view,
    vecmem::data::vector_view<const unsigned int> sizes_view,
    const int n_in_params, const int n_total_measurements,
    vecmem::data::vector_view<unsigned int> n_threads_view,
    unsigned int& n_measurements_per_thread, unsigned int& n_total_threads) {

    bound_track_parameters_collection_types::const_device params(params_view);
    vecmem::device_vector<const detray::geometry::barcode> barcodes(
        barcodes_view);
    vecmem::device_vector<const unsigned int> sizes(sizes_view);
    vecmem::device_vector<unsigned int> n_threads(n_threads_view);

    const unsigned int n_params = params.size();

    if (globalIndex >= n_in_params) {
        return;
    }

    // Get barcode
    const auto bcd = params.at(globalIndex).surface_link();

    // Search for the corresponding index of unique vector
    const auto lower =
        thrust::lower_bound(thrust::seq, barcodes.begin(), barcodes.end(), bcd);
    const auto idx = thrust::distance(barcodes.begin(), lower);

    // The averaged number of measurement per track
    const unsigned int n_avg_meas_per_track =
        (n_total_measurements + n_params - 1) / n_params;

    // Estimate how many measurements should be handled per thread in CKF
    const unsigned int n_meas_per_thread =
        (n_avg_meas_per_track + cfg.n_avg_threads_per_track - 1) /
        cfg.n_avg_threads_per_track;

    if (globalIndex == 0) {
        n_measurements_per_thread = n_meas_per_thread;
    }

    // Set the number of threads assigned per track
    n_threads.at(globalIndex) =
        (sizes.at(idx) + n_meas_per_thread - 1) / n_meas_per_thread;

    // Estimate the total number of threads we need for CKF
    vecmem::device_atomic_ref<unsigned int> num_total_threads(n_total_threads);
    num_total_threads.fetch_add(n_threads.at(globalIndex));
}

}  // namespace traccc::device
