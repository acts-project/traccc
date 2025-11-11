/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include <detray/core/detector.hpp>
#include <detray/detectors/bfield.hpp>
#include <traccc/geometry/detector.hpp>

#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/status_codes.hpp"
#include "traccc/fitting/triplet_fit/triplet_fitter.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

#include <iostream>
#include <fstream>

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
///
/// @param[in] fitter           The fitter object to use on the track candidates
/// @param[in] track_candidates All track candidates to fit
/// @param[in] mr               Memory resource to use for the output container
///
/// @return A container of the fitted track states
///
template <typename fitter_t>
track_state_container_types::host fit_tracks(
    fitter_t& fitter,
    const track_candidate_container_types::const_view& track_candidates_view,
    vecmem::memory_resource& mr, vecmem::copy& copy) {

    // Open a file
    std::ofstream file_out;
    file_out.open("/home/atlas/nandi/fit_out.csv", std::ios_base::app);

    // Create the output container.
    track_state_container_types::host result{&mr};

    // Iterate over the tracks,
    const track_candidate_container_types::const_device track_candidates{
        track_candidates_view};
    for (track_candidate_container_types::const_device::size_type i = 0;
         i < track_candidates.size(); ++i) {

        // Make a vector of track states for this track.
        vecmem::vector<track_state<typename fitter_t::algebra_type>>
            input_states{&mr};
        input_states.reserve(track_candidates.get_items()[i].size());
        for (auto& measurement : track_candidates.get_items()[i]) {
            input_states.emplace_back(measurement);
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
        kalman_fitter_status fit_status = fitter.fit(
            track_candidates.get_headers()[i].seed_params, fitter_state);

        // file_out << fitter_state.m_fit_res.fit_params.bound_local()[0] << ", " << fitter_state.m_fit_res.fit_params.bound_local()[1] << ", " << fitter_state.m_fit_res.fit_params.phi() << ", " << fitter_state.m_fit_res.fit_params.theta() << ", " << fitter_state.m_fit_res.fit_params.qop() << ", " << fitter_state.m_fit_res.fit_params.time() << ", " << fitter_state.m_fit_res.trk_quality.chi2 << ", " << fitter_state.m_fit_res.trk_quality.ndf << std::endl;

        // std::cout << "input_states[0].smoothed "  << input_states[0].smoothed().vector() << std::endl;
        // std::cout << "fit params : " << fitter_state.m_fit_res.fit_params.vector() << std::endl;

        file_out << input_states[0].smoothed()[0] << ", " << input_states[0].smoothed()[1] << ", " << input_states[0].smoothed()[2] << ", " << input_states[0].smoothed()[3] << ", " << input_states[0].smoothed()[4] << ", " << input_states[0].smoothed()[5] << ", " << fitter_state.m_fit_res.trk_quality.chi2 << ", " << fitter_state.m_fit_res.trk_quality.ndf << std::endl;

        
        file_out << getter::element(input_states[0].smoothed().covariance(), 0u, 0u) << ", " << getter::element(input_states[0].smoothed().covariance(), 1u, 1u) << ", " << getter::element(input_states[0].smoothed().covariance(), 2u, 2u) << ", " << getter::element(input_states[0].smoothed().covariance(), 3u, 3u) << ", " << getter::element(input_states[0].smoothed().covariance(), 4u, 4u) << ", " << getter::element(input_states[0].smoothed().covariance(), 5u, 5u) << std::endl;

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

/// Specialization for Triplet-based track fitting
///
/// Inlining to avoid linker error pertaining to
/// multiple definitions of the full specialization.
/// Another way out is to have the full specialization
/// in the .cpp file. Effects on performance have to be
/// studied.
///
/// @param[in] fitter           The triplet fitter object to use on the track
/// candidates
/// @param[in] track_candidates All track candidates to fit
/// @param[in] mr               Memory resource to use for the output container
///
/// @return A container of the fitted track states
///

template <>
inline track_state_container_types::host fit_tracks<>(
    traccc::triplet_fitter<
        const typename traccc::default_detector::host,
        typename detray::bfield::const_field_t<
            traccc::default_detector::host::scalar_type>::view_t>& fitter,
    const track_candidate_container_types::const_view& track_candidates_view,
    vecmem::memory_resource& mr, vecmem::copy& copy) {

    // Open a file
    std::ofstream file_out;
    file_out.open("/home/atlas/nandi/fit_out.csv", std::ios_base::app);

    // Algebra type
    using algebra_type = traccc::triplet_fitter<
        const traccc::default_detector::host,
        typename detray::bfield::const_field_t<
            traccc::default_detector::host::scalar_type>::view_t>::algebra_type;

    // Create the output container
    track_state_container_types::host result{&mr};

    // Iterate over the tracks,
    const track_candidate_container_types::const_device track_candidates{
        track_candidates_view};
    for (track_candidate_container_types::const_device::size_type i = 0;
         i < track_candidates.size(); ++i) {

        // Make a vector of track states for this track.
        vecmem::vector<track_state<algebra_type>> input_states{&mr};

        input_states.reserve(track_candidates.get_items()[i].size());
        for (auto& measurement : track_candidates.get_items()[i]) {
            input_states.emplace_back(measurement);
        }

        vecmem::data::vector_buffer<detray::geometry::barcode> seqs_buffer{};
        copy.setup(seqs_buffer)->wait();

        // Fitting result & vector of
        // fitted track states
        fitting_result<algebra_type> fit_res;
        vecmem::vector<track_state<algebra_type>> fitted_states;

        // Initialize fitter
        fitter.init_fitter(std::move(input_states));

        // Make triplets of measurements
        fitter.make_triplets();

        // Run fitter
        fitter.fit(fit_res, fitted_states);

        // Save the results into the output container.
        result.push_back(std::move(fit_res), std::move(fitted_states));

        // file_out << "track candidate " << i << std::endl;            
        
        file_out << fit_res.fit_params.bound_local()[0] << ", " << fit_res.fit_params.bound_local()[1] << ", " << fit_res.fit_params.phi() << ", " << fit_res.fit_params.theta() << ", " << fit_res.fit_params.qop() << ", " << fit_res.fit_params.time() << ", " << fit_res.trk_quality.chi2 << ", " << fit_res.trk_quality.ndf << ", " << fit_res.c_3D << ", " << fit_res.sig_c_3D << std::endl;
        
        file_out << getter::element(fit_res.fit_params.covariance(), 0u, 0u) << ", " << getter::element(fit_res.fit_params.covariance(), 1u, 1u) << ", " << getter::element(fit_res.fit_params.covariance(), 2u, 2u) << ", " << getter::element(fit_res.fit_params.covariance(), 3u, 3u) << ", " << getter::element(fit_res.fit_params.covariance(), 4u, 4u) << ", " << getter::element(fit_res.fit_params.covariance(), 5u, 5u) << std::endl;
    }

    // file_out.close();

    // Return the fitted track states.
    return result;
}

}  // namespace traccc::host::details
