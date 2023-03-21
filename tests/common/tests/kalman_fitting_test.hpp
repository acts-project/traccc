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
#include <detray/core/detector.hpp>
#include <detray/detectors/detector_metadata.hpp>
#include <detray/propagator/navigator.hpp>
#include <detray/propagator/rk_stepper.hpp>

#include "detray/masks/unbounded.hpp"

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <string>
#include <string_view>
#include <vector>

namespace traccc {

/// Kalman Fitting Test with Telescope Geometry
///
/// Parameter for data directory
class KalmanFittingTests
    : public ::testing::TestWithParam<
          std::tuple<std::string, unsigned int, unsigned int>> {

    public:
    /// Type declarations
    using host_detector_type =
        detray::detector<detray::detector_registry::template telescope_detector<
                             detray::unbounded<detray::rectangle2D<>>>,
                         covfie::field, detray::host_container_types>;
    using device_detector_type =
        detray::detector<detray::detector_registry::template telescope_detector<
                             detray::unbounded<detray::rectangle2D<>>>,
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