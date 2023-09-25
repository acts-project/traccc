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

/// Function evalulating the number of measurements to be iterated per thread
/// and the total number of threads required
///
/// @param[in] globalIndex           The index of the current thread
/// @param[in] cfg                   Track finding config object
/// @param[in] params_view           Input parameters view object
/// @param[in] barcodes_view         Barcodes view object
/// @param[in] sizes_view            The number of measurements of each module
/// @param[in] n_in_params           The number of input track parameters
/// @param[in] n_total_measurements  Total number of meausurments
/// @param[out] n_threads_view       The number of threads per tracks
/// @param[out] n_measurements_per_thread  The number of measurements to be
/// iterated per thread
/// @param[out] n_total_thread       Total number of threads
///
template <typename config_t>
TRACCC_DEVICE inline void count_threads(
    std::size_t globalIndex, const config_t cfg,
    bound_track_parameters_collection_types::const_view params_view,
    vecmem::data::vector_view<const detray::geometry::barcode> barcodes_view,
    vecmem::data::vector_view<const unsigned int> sizes_view,
    const int n_in_params, const int n_total_measurements,
    vecmem::data::vector_view<unsigned int> n_threads_view,
    unsigned int& n_measurements_per_thread, unsigned int& n_total_threads);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/finding/device/impl/count_threads.ipp"