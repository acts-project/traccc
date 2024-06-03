/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_DEVICE inline void count_measurements(
    std::size_t globalIndex,
    bound_track_parameters_collection_types::const_view params_view,
    vecmem::data::vector_view<const detray::geometry::barcode> barcodes_view,
    vecmem::data::vector_view<const unsigned int> upper_bounds_view,
    const unsigned int n_in_params,
    vecmem::data::vector_view<unsigned int> n_measurements_view,
    vecmem::data::vector_view<unsigned int> ref_meas_idx_view,
    unsigned int& n_measurements_sum) {

    bound_track_parameters_collection_types::const_device params(params_view);
    vecmem::device_vector<const detray::geometry::barcode> barcodes(
        barcodes_view);
    vecmem::device_vector<const unsigned int> upper_bounds(upper_bounds_view);
    vecmem::device_vector<unsigned int> n_measurements(n_measurements_view);
    vecmem::device_vector<unsigned int> ref_meas_idx(ref_meas_idx_view);

    if (globalIndex >= n_in_params) {
        return;
    }

    // Get barcode
    const auto bcd = params.at(globalIndex).surface_link();
    const auto lo =
        thrust::lower_bound(thrust::seq, barcodes.begin(), barcodes.end(), bcd);

    // If barcode is not found (no measurement)
    if (lo == barcodes.end()) {
        return;
    }

    const auto bcd_id = std::distance(barcodes.begin(), lo);

    // Get the reference measurement index and the number of measurements per
    // parameter
    ref_meas_idx.at(globalIndex) =
        lo == barcodes.begin() ? 0u : upper_bounds[bcd_id - 1];
    n_measurements.at(globalIndex) =
        upper_bounds[bcd_id] - ref_meas_idx.at(globalIndex);

    // Increase the total number of measurements with atomic addition
    vecmem::device_atomic_ref<unsigned int> n_meas_sum(n_measurements_sum);
    n_meas_sum.fetch_add(n_measurements.at(globalIndex));
}

}  // namespace traccc::device
