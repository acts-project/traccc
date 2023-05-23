/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/utils/compare.hpp"

// Thrust include(s).
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace traccc::device {

template <typename detector_t>
TRACCC_DEVICE inline void count_measurements(
    std::size_t globalIndex, typename detector_t::detector_view_type det_data,
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const thrust::pair<geometry_id, unsigned int>>
        module_map_view,
    const int n_params,
    bound_track_parameters_collection_types::const_view params_view,
    vecmem::data::vector_view<unsigned int> n_measurements_view,
    unsigned int& n_total_measurements) {

    // Detector
    detector_t det(det_data);

    // Measurement
    measurement_container_types::const_device measurements(measurements_view);

    // module map
    vecmem::device_vector<const thrust::pair<geometry_id, unsigned int>>
        module_map(module_map_view);

    // N measurements
    vecmem::device_vector<unsigned int> n_measurements(n_measurements_view);

    // Parameters
    bound_track_parameters_collection_types::const_device params(params_view);

    if (globalIndex >= n_params) {
        return;
    }

    // Get module id
    const auto module_id = params.at(globalIndex).surface_link();

    // Search for measuremetns header ID
    const auto lower = thrust::lower_bound(
        thrust::seq, module_map.begin(), module_map.end(), module_id.value(),
        compare_pair_int<thrust::pair, unsigned int>());
    const auto header_id = (*lower).second;

    n_measurements.at(globalIndex) = measurements.at(header_id).items.size();

    vecmem::device_atomic_ref<unsigned int> num_total_measurements(
        n_total_measurements);
    num_total_measurements.fetch_add(n_measurements.at(globalIndex));
}

}  // namespace traccc::device
