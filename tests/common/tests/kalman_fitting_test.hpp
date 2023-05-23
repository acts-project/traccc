/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"

// detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/detector_metadata.hpp"
#include "detray/masks/masks.hpp"
#include "detray/propagator/navigator.hpp"
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
/// (7) number of tracks per event
/// (8) number of events
class KalmanFittingTests
    : public ::testing::TestWithParam<std::tuple<
          std::string, std::array<scalar, 3u>, std::array<scalar, 3u>,
          std::array<scalar, 2u>, std::array<scalar, 2u>,
          std::array<scalar, 2u>, unsigned int, unsigned int>> {

    public:
    /// Type declarations
    using host_detector_type =
        detray::detector<detray::detector_registry::template telescope_detector<
                             detray::rectangle2D<>>,
                         covfie::field, detray::host_container_types>;
    using device_detector_type =
        detray::detector<detray::detector_registry::template telescope_detector<
                             detray::rectangle2D<>>,
                         covfie::field_view, detray::device_container_types>;

    using b_field_t = typename host_detector_type::bfield_type;
    using rk_stepper_type = detray::rk_stepper<b_field_t::view_t, transform3,
                                               detray::constrained_step<>>;
    using host_navigator_type = detray::navigator<const host_detector_type>;
    using host_fitter_type =
        kalman_fitter<rk_stepper_type, host_navigator_type>;
    using device_navigator_type = detray::navigator<const device_detector_type>;
    using device_fitter_type =
        kalman_fitter<rk_stepper_type, device_navigator_type>;

    // Use deterministic random number generator for testing
    using uniform_gen_t =
        detray::random_numbers<scalar, std::uniform_real_distribution<scalar>,
                               std::seed_seq>;

    /// Plane alignment direction (aligned to x-axis)
    static const inline detray::detail::ray<transform3> traj{
        {0, 0, 0}, 0, {1, 0, 0}, -1};

    /// Position of planes (in mm unit)
    static const inline std::vector<scalar> plane_positions = {
        20., 40., 60., 80., 100., 120., 140, 160, 180.};

    /// B field value and its type
    static constexpr vector3 B{2 * detray::unit<scalar>::T, 0, 0};

    /// Plane material and thickness
    static const inline detray::silicon_tml<scalar> mat = {};
    static constexpr scalar thickness = 0.5 * detray::unit<scalar>::mm;

    // Rectangle mask for the telescope geometry
    static constexpr detray::mask<detray::rectangle2D<>> rectangle{0u, 100000.f,
                                                                   100000.f};

    /// Measurement smearing parameters
    static constexpr std::array<scalar, 2u> smearing{
        50 * detray::unit<scalar>::um, 50 * detray::unit<scalar>::um};

    /// Standard deviations for seed track parameters
    static constexpr std::array<scalar, e_bound_size> stddevs = {
        0.03 * detray::unit<scalar>::mm,
        0.03 * detray::unit<scalar>::mm,
        0.017,
        0.017,
        0.001 / detray::unit<scalar>::GeV,
        1 * detray::unit<scalar>::ns};

    /// Verify that pull distribtions follow the normal distribution
    ///
    /// @param file_name The name of the file holding the distributions
    /// @param hist_names The names of the histograms to process
    ///
    void pull_value_tests(std::string_view file_name,
                          const std::vector<std::string>& hist_names) const;
};

}  // namespace traccc