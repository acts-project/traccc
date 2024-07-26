/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

// Thrust include(s)
#include <thrust/binary_search.h>

// System include(s)
#include <iterator>

namespace traccc::device {

/// Function evalulating the number of measurements to be iterated per thread
/// and the total number of measurements
///
/// @param[in] globalIndex           The index of the current thread
/// @param[in] params_view           Input parameters view object
/// @param[in] barcodes_view         Barcodes view object
/// @param[in] upper_bounds          Upper bounds of measurements w.r.t geometry
/// ID
/// @param[out] n_measurements_view  The number of measurements per parameter
/// @param[out] ref_meas_idx         The first index of measurements per
/// parameter
/// @param[out] n_measurements_sum   The sum of the number of measurements per
/// parameter
///
TRACCC_DEVICE inline void count_measurements(
    std::size_t globalIndex,
    bound_track_parameters_collection_types::const_view params_view,
    vecmem::data::vector_view<const detray::geometry::barcode> barcodes_view,
    vecmem::data::vector_view<const unsigned int> upper_bounds_view,
    const unsigned int n_in_params,
    vecmem::data::vector_view<unsigned int> n_measurements_view,
    vecmem::data::vector_view<unsigned int> ref_meas_idx_view,
    unsigned int& n_measurements_sum);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/finding/device/impl/count_measurements.ipp"
