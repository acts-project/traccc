/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate_collection.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/status_codes.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

namespace traccc::host::details {

/// Templated implementation of the track fitting algorithm.
///
/// Concrete track fitting algorithms can use this function with the appropriate
/// specializations, to fit tracks on top of a specific detector type, magnetic
/// field type, and track fitting configuration.
///
/// @note The memory resource received by this function is not used thoroughly
///       for the setup of the output container. Inner vectors in the output's
///       jagged vector are created using the default memory resource.
///
/// @tparam fitter_t The fitter type used for the track fitting
/// @tparam algebra_t The algebra type used for the track fitting
///
/// @param[in] fitter           The fitter object to use on the track candidates
/// @param[in] track_candidates All track candidates to fit
/// @param[in] mr               Memory resource to use for the output container
///
/// @return A container of the fitted track states
///
template <typename algebra_t, typename fitter_t>
track_state_container_types::host fit_tracks(
    fitter_t& fitter,
    const measurement_collection_types::const_view& measurements_view,
    const typename edm::track_candidate_collection<algebra_t>::const_view&
        track_candidates_view,
    vecmem::memory_resource& mr, vecmem::copy& copy) {

    // Create the input container(s).
    const measurement_collection_types::const_device measurements{
        measurements_view};
    const typename edm::track_candidate_collection<algebra_t>::const_device
        track_candidates{track_candidates_view};

    // Create the output container.
    track_state_container_types::host result{&mr};

    // Iterate over the tracks,
    for (typename edm::track_candidate_collection<
             algebra_t>::const_device::size_type i = 0;
         i < track_candidates.size(); ++i) {

        // Make a vector of track states for this track.
        vecmem::vector<track_state<typename fitter_t::algebra_type> >
            input_states{&mr};
        input_states.reserve(
            track_candidates.measurement_indices().at(i).size());
        for (unsigned int measurement_index :
             track_candidates.measurement_indices().at(i)) {
            input_states.emplace_back(measurements.at(measurement_index));
        }

        vecmem::data::vector_buffer<detray::geometry::barcode> seqs_buffer{
            static_cast<vecmem::data::vector_buffer<
                detray::geometry::barcode>::size_type>(
                std::max(input_states.size() *
                             fitter.config().barcode_sequence_size_factor,
                         fitter.config().min_barcode_sequence_capacity)),
            mr, vecmem::data::buffer_type::resizable};
        copy.setup(seqs_buffer)->wait();

        // Make a fitter state
        typename fitter_t::state fitter_state(vecmem::get_data(input_states),
                                              seqs_buffer);

        // Run the fitter.
        kalman_fitter_status fit_status =
            fitter.fit(track_candidates.params().at(i), fitter_state);

        if (fit_status == kalman_fitter_status::SUCCESS) {
            // Save the results into the output container.
            result.push_back(std::move(fitter_state.m_fit_res),
                             std::move(input_states));
        } else {
            // TODO: Print a warning here.
        }
    }

    // Return the fitted track states.
    return result;
}

}  // namespace traccc::host::details
