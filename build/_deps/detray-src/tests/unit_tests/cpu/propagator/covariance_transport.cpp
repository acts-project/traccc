/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/definitions/units.hpp"
#include "detray/geometry/detail/surface_descriptor.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes.hpp"
#include "detray/geometry/shapes/unbounded.hpp"
#include "detray/navigation/detail/ray.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/line_stepper.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/tracks/tracks.hpp"
#include "detray/utils/axis_rotation.hpp"

// Detray test include(s)
#include "detray/test/utils/detectors/build_telescope_detector.hpp"
#include "detray/test/utils/types.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// google-test include(s).
#include <gtest/gtest.h>

using namespace detray;

// Algebra types
using algebra_t = test::algebra;
using point2 = test::point2;
using vector3 = test::vector3;
using matrix_operator = test::matrix_operator;

// Mask types to be tested
// @TODO: Remove unbounded tag
using annulus_type = detray::mask<detray::unbounded<detray::annulus2D>>;
using rectangle_type = detray::mask<detray::rectangle2D>;
using trapezoid_type = detray::mask<detray::trapezoid2D>;
using ring_type = detray::mask<detray::ring2D>;
using cylinder_type = detray::mask<detray::cylinder2D>;
using straw_tube_type = detray::mask<detray::line_circular>;
using drift_cell_type = detray::mask<detray::line_square>;

constexpr scalar tol{1e-6f};

GTEST_TEST(detray_propagator, covariance_transport) {

    vecmem::host_memory_resource host_mr;

    // Build in x-direction from given module positions
    detail::ray<algebra_t> traj{{0.f, 0.f, 0.f}, 0.f, {1.f, 0.f, 0.f}, -1.f};
    std::vector<scalar> positions = {0.f, 10.f, 20.f, 30.f, 40.f, 50.f, 60.f};

    tel_det_config<rectangle2D> tel_cfg{200.f * unit<scalar>::mm,
                                        200.f * unit<scalar>::mm};
    tel_cfg.positions(positions).pilot_track(traj);

    // Build telescope detector with unbounded planes
    const auto [det, names] = build_telescope_detector(host_mr, tel_cfg);

    using navigator_t = navigator<decltype(det)>;
    using cline_stepper_t = line_stepper<algebra_t>;
    using actor_chain_t = actor_chain<dtuple, parameter_transporter<algebra_t>,
                                      parameter_resetter<algebra_t>>;
    using propagator_t =
        propagator<cline_stepper_t, navigator_t, actor_chain_t>;

    // Bound vector
    bound_parameters_vector<algebra_t> bound_vector{};
    bound_vector.set_theta(constant<scalar>::pi_4);
    bound_vector.set_qop(-0.1f);

    // Bound covariance
    typename bound_track_parameters<algebra_t>::covariance_type bound_cov =
        matrix_operator().template identity<e_bound_size, e_bound_size>();

    // Note: Set angle error as ZERO, to constrain the loc0 and loc1 divergence
    getter::element(bound_cov, e_bound_phi, e_bound_phi) = 0.f;
    getter::element(bound_cov, e_bound_theta, e_bound_theta) = 0.f;

    // Bound track parameter
    const bound_track_parameters<algebra_t> bound_param0(
        geometry::barcode{}.set_index(0u), bound_vector, bound_cov);

    // Actors
    parameter_transporter<algebra_t>::state bound_updater{};
    parameter_resetter<algebra_t>::state rst{};

    propagation::config prop_cfg{};
    prop_cfg.navigation.overstep_tolerance = -100.f * unit<float>::um;
    propagator_t p{prop_cfg};
    propagator_t::state propagation(bound_param0, det, prop_cfg.context);

    // Run propagator
    p.propagate(propagation, detray::tie(bound_updater, rst));

    // Bound state after one turn propagation
    const auto& bound_param1 = propagation._stepping.bound_params();

    // Check if the track reaches the final surface
    EXPECT_EQ(bound_param0.surface_link().volume(), 4095u);
    EXPECT_EQ(bound_param0.surface_link().index(), 0u);
    EXPECT_EQ(bound_param1.surface_link().volume(), 0u);
    EXPECT_EQ(bound_param1.surface_link().id(), surface_id::e_sensitive);
    EXPECT_EQ(bound_param1.surface_link().index(), 6u);

    const auto bound_cov0 = bound_param0.covariance();
    const auto bound_cov1 = bound_param1.covariance();

    // Check covaraince
    for (unsigned int i = 0u; i < e_bound_size; i++) {
        for (unsigned int j = 0u; j < e_bound_size; j++) {
            EXPECT_NEAR(matrix_operator().element(bound_cov0, i, j),
                        matrix_operator().element(bound_cov1, i, j), tol);
        }
    }
}
