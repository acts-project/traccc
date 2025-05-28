/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c traccc::device::count_shared_measurements
/// function
struct count_shared_measurements_payload {

    /**
     * @brief View object to the accepted ids
     */
    vecmem::data::vector_view<const unsigned int> accepted_ids_view;

    /**
     * @brief View object to the vector of measured ids per track
     */
    vecmem::data::jagged_vector_view<const std::size_t> meas_ids_view;

    /**
     * @brief View object to the unique measurement ids
     */
    vecmem::data::vector_view<const std::size_t> unique_meas_view;

    /**
     * @brief View object to the tracks per measurement
     */
    vecmem::data::vector_view<const unsigned int>
        n_accepted_tracks_per_measurement_view;

    /**
     * @brief View object to the number of shared measurements
     */
    vecmem::data::vector_view<unsigned int> n_shared_view;
};

}  // namespace traccc::device
