/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// detray include(s)
#include "detray/propagator/rk_stepper.hpp"

#include "detray/builders/volume_builder.hpp"
#include "detray/core/detector.hpp"
#include "detray/definitions/units.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/geometry/tracking_surface.hpp"
#include "detray/io/utils/file_handle.hpp"
#include "detray/navigation/detail/trajectories.hpp"
#include "detray/propagator/line_stepper.hpp"
#include "detray/propagator/stepping_config.hpp"
#include "detray/tracks/tracks.hpp"

// Detray test include(s)
#include "detray/test/utils/simulation/event_generator/track_generators.hpp"
#include "detray/test/utils/types.hpp"

// System include(s)
#include <memory>

// google-test include(s)
#include <gtest/gtest.h>

using namespace detray;

using algebra_t = test::algebra;
using scalar_t = test::scalar;
using vector3 = test::vector3;
using point3 = test::point3;
using matrix_operator = test::matrix_operator;

/// Runge-Kutta stepper
template <typename bfield_t>
using rk_stepper_t = rk_stepper<typename bfield_t::view_t, algebra_t>;
template <typename bfield_t>
using crk_stepper_t =
    rk_stepper<typename bfield_t::view_t, algebra_t, constrained_step<>>;

namespace {

constexpr scalar_t tol{1e-3f};

const stepping::config step_cfg{};
constexpr scalar_t step_size{1.f * unit<scalar_t>::mm};
constexpr material<scalar_t> vol_mat{
    detray::cesium_iodide_with_ded<scalar_t>()};

}  // namespace

// This tests the base functionality of the Runge-Kutta stepper
GTEST_TEST(detray_propagator, rk_stepper) {
    using namespace step;

    // Constant magnetic field
    using bfield_t = bfield::const_field_t;

    vector3 B{1.f * unit<scalar_t>::T, 1.f * unit<scalar_t>::T,
              1.f * unit<scalar_t>::T};
    const bfield_t hom_bfield = bfield::create_const_field(B);

    // RK stepper
    rk_stepper_t<bfield_t> rk_stepper;
    crk_stepper_t<bfield_t> crk_stepper;

    // RK stepper configurations
    constexpr unsigned int rk_steps = 100u;
    constexpr scalar_t stepsize_constr{0.5f * unit<scalar_t>::mm};

    // Track generator configuration
    const scalar_t p_mag{10.f * unit<scalar_t>::GeV};
    constexpr unsigned int theta_steps = 100u;
    constexpr unsigned int phi_steps = 100u;

    // Iterate through uniformly distributed momentum directions
    for (auto track : uniform_track_generator<free_track_parameters<algebra_t>>(
             phi_steps, theta_steps, p_mag)) {
        // Generate track state used for propagation with constrained step size
        free_track_parameters<algebra_t> c_track(track);

        // helix trajectory
        detail::helix helix(track, &B);

        // RK Stepping into forward direction
        rk_stepper_t<bfield_t>::state rk_state{track, hom_bfield};
        crk_stepper_t<bfield_t>::state crk_state{c_track, hom_bfield};

        // Set step size constraint to half the nominal step size =>
        // crk_stepper will need twice as many steps
        crk_state.template set_constraint<constraint::e_user>(stepsize_constr);
        ASSERT_NEAR(crk_state.constraints().template size<>(),
                    0.5f * unit<scalar_t>::mm, tol);

        // Forward stepping
        for (unsigned int i_s = 0u; i_s < rk_steps; i_s++) {
            rk_stepper.step(step_size, rk_state, step_cfg, true);
            crk_stepper.step(step_size, crk_state, step_cfg, true);
            crk_stepper.step(step_size, crk_state, step_cfg, true);
        }

        // Check that both steppers arrive at the same point
        // Get relative error by dividing error with path length
        ASSERT_TRUE(rk_state.path_length() > 0.f);
        ASSERT_NEAR(rk_state.path_length(), crk_state.path_length(), tol);
        ASSERT_NEAR(getter::norm(rk_state().pos() - crk_state().pos()) /
                        rk_state.path_length(),
                    0.f, tol);

        // Check that the stepper position lies on the truth helix
        const auto helix_pos = helix(rk_state.path_length());
        const auto forward_pos = rk_state().pos();
        const point3 forward_relative_error{(1.f / rk_state.path_length()) *
                                            (forward_pos - helix_pos)};

        // Make sure that relative error is smaller than the tolerance
        EXPECT_NEAR(getter::norm(forward_relative_error), 0.f, tol);

        // Roll the same track back to the origin
        // Use the same path length, since there is no overstepping
        const scalar_t path_length = rk_state.path_length();
        for (unsigned int i_s = 0u; i_s < rk_steps; i_s++) {
            rk_stepper.step(-step_size, rk_state, step_cfg, true);
            crk_stepper.step(-step_size, crk_state, step_cfg, true);
            crk_stepper.step(-step_size, crk_state, step_cfg, true);
        }

        // Should arrive back at track origin, where path length is zero
        ASSERT_NEAR(rk_state.path_length(), 0.f, tol);
        ASSERT_NEAR(crk_state.path_length(), 0.f, tol);

        const point3 backward_relative_error{1.f / (2.f * path_length) *
                                             (rk_state().pos())};
        // Make sure that relative error is smaller than the tolerance
        EXPECT_NEAR(getter::norm(backward_relative_error), 0.f, tol);

        // The constrained stepper should be at the same position now
        ASSERT_NEAR(getter::norm(rk_state().pos() - crk_state().pos()) /
                        (2.f * path_length),
                    0.f, tol);
    }
}

/// This tests the base functionality of the Runge-Kutta stepper in an
/// in-homogeneous magnetic field, read from file
TEST(detray_propagator, rk_stepper_inhomogeneous_bfield) {
    using namespace step;

    // Read the magnetic field map
    using bfield_t = bfield::inhom_field_t;
    bfield_t inhom_bfield = bfield::create_inhom_field();

    // RK stepper
    rk_stepper_t<bfield_t> rk_stepper;
    crk_stepper_t<bfield_t> crk_stepper;

    // RK stepper configurations
    constexpr unsigned int rk_steps = 100u;
    constexpr scalar_t stepsize_constr{0.5f * unit<scalar_t>::mm};

    // Track generator configuration
    const scalar_t p_mag{10.f * unit<scalar_t>::GeV};
    constexpr unsigned int theta_steps = 100u;
    constexpr unsigned int phi_steps = 100u;

    // Iterate through uniformly distributed momentum directions
    for (auto track : uniform_track_generator<free_track_parameters<algebra_t>>(
             phi_steps, theta_steps, p_mag)) {
        // Generate track state used for propagation with constrained step size
        free_track_parameters<algebra_t> c_track(track);

        // RK Stepping into forward direction
        rk_stepper_t<bfield_t>::state rk_state{track, inhom_bfield};
        crk_stepper_t<bfield_t>::state crk_state{c_track, inhom_bfield};

        crk_state.template set_constraint<constraint::e_user>(stepsize_constr);
        ASSERT_NEAR(crk_state.constraints().template size<>(),
                    0.5f * unit<scalar_t>::mm, tol);

        // Forward stepping
        for (unsigned int i_s = 0u; i_s < rk_steps; i_s++) {
            rk_stepper.step(step_size, rk_state, step_cfg, true);
            crk_stepper.step(step_size, crk_state, step_cfg, true);
            crk_stepper.step(step_size, crk_state, step_cfg, true);
        }

        // Make sure the steppers moved
        const scalar_t path_length{rk_state.path_length()};
        ASSERT_TRUE(path_length > 0.f);
        ASSERT_TRUE(crk_state.path_length() > 0.f);

        // Roll the same track back to the origin
        // Use the same path length, since there is no overstepping
        for (unsigned int i_s = 0u; i_s < rk_steps; i_s++) {
            rk_stepper.step(-step_size, rk_state, step_cfg, true);
            crk_stepper.step(-step_size, crk_state, step_cfg, true);
            crk_stepper.step(-step_size, crk_state, step_cfg, true);
        }

        // Should arrive back at track origin, where path length is zero
        ASSERT_NEAR(rk_state.path_length(), 0.f, tol);
        ASSERT_NEAR(crk_state.path_length(), 0.f, tol);

        const point3 backward_relative_error{1.f / (2.f * path_length) *
                                             (rk_state().pos())};
        // Make sure that relative error is smaller than the tolerance
        EXPECT_NEAR(getter::norm(backward_relative_error), 0.f, tol);

        // The constrained stepper should be at the same position now
        ASSERT_NEAR(getter::norm(rk_state().pos() - crk_state().pos()) /
                        (2.f * path_length),
                    0.f, tol);
    }
}

/// This tests dqop of the Runge-Kutta stepper
TEST(detray_propagator, qop_derivative) {
    using namespace step;

    // Constant magnetic field
    using bfield_t = bfield::const_field_t;

    vector3 B{0.f * unit<scalar_t>::T, 0.f * unit<scalar_t>::T,
              2.f * unit<scalar_t>::T};
    const bfield_t hom_bfield = bfield::create_const_field(B);

    // RK stepper
    rk_stepper_t<bfield_t> rk_stepper;
    constexpr unsigned int rk_steps = 1000u;

    // Theta phi for track generator
    const scalar_t p_mag{10.f * unit<scalar_t>::GeV};
    constexpr unsigned int theta_steps = 10u;
    constexpr unsigned int phi_steps = 10u;

    const scalar_t ds = 1e-2f * unit<scalar_t>::mm;

    // Iterate through uniformly distributed momentum directions
    for (auto track : uniform_track_generator<free_track_parameters<algebra_t>>(
             phi_steps, theta_steps, p_mag)) {

        // RK Stepping into forward direction
        rk_stepper_t<bfield_t>::state rk_state{track, hom_bfield};

        for (unsigned int i_s = 0u; i_s < rk_steps; i_s++) {

            const scalar_t qop1 = rk_state().qop();
            const scalar_t d2qopdsdqop = rk_state.d2qopdsdqop(qop1, &vol_mat);

            const scalar_t dqopds1 = rk_state.dqopds(qop1, &vol_mat);

            rk_stepper.step(ds, rk_state, step_cfg, true, &vol_mat);

            const scalar_t qop2 = rk_state().qop();
            const scalar_t dqopds2 = rk_state.dqopds(qop2, &vol_mat);

            ASSERT_TRUE(qop1 > qop2);
            ASSERT_NEAR((qop2 - qop1) / ds, dqopds1, 1e-4);
            ASSERT_NEAR((dqopds2 - dqopds1) / (qop2 - qop1), d2qopdsdqop, 1e-4);
        }
    }
}
