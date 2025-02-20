/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/geometry/detector.hpp"

// detray include(s).
#include <detray/definitions/pdg_particle.hpp>
#include <detray/detectors/bfield.hpp>
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/propagator.hpp>
#include <detray/propagator/rk_stepper.hpp>
#include <detray/test/utils/simulation/event_generator/track_generators.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <string>
#include <string_view>
#include <vector>

namespace traccc {

class KalmanFittingTests : public testing::Test {
    public:
    /// Type declarations
    using host_detector_type = traccc::default_detector::host;
    using device_detector_type = traccc::default_detector::device;

    using scalar_type = device_detector_type::scalar_type;
    using b_field_t = covfie::field<detray::bfield::const_bknd_t<scalar_type>>;
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t, traccc::default_algebra,
                           detray::constrained_step<scalar_type>>;
    using host_navigator_type = detray::navigator<const host_detector_type>;
    using host_fitter_type =
        kalman_fitter<rk_stepper_type, host_navigator_type>;
    using device_navigator_type = detray::navigator<const device_detector_type>;
    using device_fitter_type =
        kalman_fitter<rk_stepper_type, device_navigator_type>;

    // Use deterministic random number generator for testing
    using uniform_gen_t =
        detray::detail::random_numbers<scalar,
                                       std::uniform_real_distribution<scalar>>;

    /// Verify that pull distribtions follow the normal distribution
    ///
    /// @param file_name The name of the file holding the distributions
    /// @param hist_names The names of the histograms to process
    ///
    void pull_value_tests(std::string_view file_name,
                          const std::vector<std::string>& hist_names) const;

    /// Verify that P value distribtions follow the uniform
    ///
    /// @param file_name The name of the file holding the distributions
    ///
    void p_value_tests(std::string_view file_name) const;

    /// Validadte the NDF for track finding output
    ///
    /// @param find_res Finding result of a track
    /// @param track_candidates_per_track Track candidates of a track
    ///
    void ndf_tests(const finding_result& find_res,
                   const track_candidate_collection_types::host&
                       track_candidates_per_track);

    /// Validadte the NDF for track fitting output
    ///
    /// @param fit_res Fitting statistics result of a track
    /// @param track_states_per_track Track states of a track
    ///
    void ndf_tests(
        const fitting_result<traccc::default_algebra>& fit_res,
        const track_state_collection_types::host& track_states_per_track);

    // The number of tracks successful with KF
    std::size_t n_success{0u};
};

}  // namespace traccc
