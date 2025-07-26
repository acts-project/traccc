/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/utils/pair.hpp"

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c
/// traccc::device::count_removable_tracks function
struct count_removable_tracks_payload {

    /**
     * @brief Whether to terminate the calculation
     */
    int* terminate;

    /**
     * @brief The number of max shared
     */
    unsigned int* max_shared;

    /**
     * @brief View object to the sorted track
     */
    vecmem::data::vector_view<const unsigned int> sorted_ids_view;

    /**
     * @brief The number of accepted tracks
     */
    unsigned int* n_accepted;

    /**
     * @brief View object to the vector of measured ids per track
     */
    vecmem::data::jagged_vector_view<const measurement_id_type> meas_ids_view;

    /**
     * @brief View object to the vector of number of measurements
     */
    vecmem::data::vector_view<const unsigned int> n_meas_view;

    /**
     * @brief View object to the unique measurement ids
     */
    vecmem::data::vector_view<const measurement_id_type> unique_meas_view;

    /**
     * @brief View object to the meas id to unique id map
     */
    vecmem::data::vector_view<const unsigned int> meas_id_to_unique_id_view;

    /**
     * @brief View object to the number of accepted tracks per measurement
     */
    vecmem::data::vector_view<const unsigned int>
        n_accepted_tracks_per_measurement_view;

    /**
     * @brief The number of worst tracks removable
     */
    unsigned int* n_removable_tracks;

    /**
     * @brief The number of measurements to remove
     */
    unsigned int* n_meas_to_remove;

    /**
     * @brief View object to measurements to remove
     */
    vecmem::data::vector_view<measurement_id_type> meas_to_remove_view;

    /**
     * @brief View object to thread id of measurements to remove
     */
    vecmem::data::vector_view<unsigned int> threads_view;

    /**
     * @brief The number of threads that can remove its corresponding track
     */
    unsigned int* n_valid_threads;
};

}  // namespace traccc::device
