/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// detray include(s).
#include "detray/detectors/detector_metadata.hpp"
#include "detray/propagator/rk_stepper.hpp"

// GTest include(s).
#include <gtest/gtest.h>

// ROOT include(s).
#include <TF1.h>
#include <TFile.h>

namespace traccc {

/// Kalman Fitting Test with Telescope Geometry
///
/// Tuple parameter made of (1) initial particle momentum and (2) initial phi
class KalmanFittingTests
    : public ::testing::TestWithParam<std::tuple<scalar, scalar>> {

    public:
    /// Type declarations
    using host_detector_type =
        detray::detector<detray::detector_registry::telescope_detector,
                         covfie::field, detray::host_container_types>;
    using device_detector_type =
        detray::detector<detray::detector_registry::telescope_detector,
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
        -10., 20., 40., 60., 80., 100., 120., 140, 160, 180., 200.};
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

    /// Verify that pull distribtion follows the normal distribution
    ///
    /// @param pull_dist the input histogram of pull values
    void pull_value_test(TH1F* pull_dist) const {
        TF1 gaus{"gaus", "gaus", -5, 5};
        double fit_par[3];

        // Set the mean seed to 0
        gaus.SetParameters(1, 0.);
        gaus.SetParLimits(1, -1., 1.);
        // Set the standard deviation seed to 1
        gaus.SetParameters(2, 1.0);
        gaus.SetParLimits(2, 0.5, 2.);

        auto res = pull_dist->Fit("gaus", "Q0S");

        gaus.GetParameters(&fit_par[0]);

        // Mean check
        EXPECT_NEAR(fit_par[1], 0, 0.05)
            << pull_dist->GetName() << " mean value error";

        // Sigma check
        EXPECT_NEAR(fit_par[2], 1, 0.1)
            << pull_dist->GetName() << " sigma value error";
    }
};

}  // namespace traccc