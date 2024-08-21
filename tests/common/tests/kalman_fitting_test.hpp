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

// detray include(s).
#include "detray/core/detector.hpp"
#include "detray/core/detector_metadata.hpp"
#include "detray/definitions/pdg_particle.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"
#include "detray/simulation/event_generator/track_generators.hpp"

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <string>
#include <string_view>
#include <vector>

namespace traccc {

/// Kalman Fitting Test with Telescope Geometry
///
/// Test parameters:
/// (1) name
/// (2) origin
/// (3) origin stddev
/// (4) momentum range
/// (5) eta range
/// (6) phi range
/// (7) particle type
/// (8) number of tracks per event
/// (9) number of events
/// (10) random charge
class KalmanFittingTests
    : public ::testing::TestWithParam<std::tuple<
          std::string, std::array<scalar, 3u>, std::array<scalar, 3u>,
          std::array<scalar, 2u>, std::array<scalar, 2u>,
          std::array<scalar, 2u>, detray::pdg_particle<scalar>, unsigned int,
          unsigned int, bool>> {

    public:
    /// Type declarations
    using host_detector_type = detray::detector<detray::default_metadata,
                                                detray::host_container_types>;
    using device_detector_type =
        detray::detector<detray::default_metadata,
                         detray::device_container_types>;

    using b_field_t = covfie::field<detray::bfield::const_bknd_t>;
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t, traccc::default_algebra,
                           detray::constrained_step<>>;
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

    /// Validadte the NDF
    ///
    /// @param host_det Detector object
    /// @param fit_res Fitting statistics result of a track
    /// @param track_candidates_per_track Track candidates of a track
    /// @param track_states_per_track Track states of a track
    ///
    void ndf_tests(
        const fitting_result<traccc::default_algebra>& fit_res,
        const track_state_collection_types::host& track_states_per_track);

    // The number of tracks successful with KF
    std::size_t n_success{0u};
};

}  // namespace traccc
