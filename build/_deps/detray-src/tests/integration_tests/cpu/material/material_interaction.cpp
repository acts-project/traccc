/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/definitions/pdg_particle.hpp"
#include "detray/definitions/units.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes/unbounded.hpp"
#include "detray/materials/interaction.hpp"
#include "detray/materials/material.hpp"
#include "detray/materials/material_slab.hpp"
#include "detray/materials/predefined_materials.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/line_stepper.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// Detray test include(s)
#include "detray/test/utils/detectors/build_telescope_detector.hpp"
#include "detray/test/utils/inspectors.hpp"
#include "detray/test/utils/simulation/random_scatterer.hpp"
#include "detray/test/utils/statistics.hpp"
#include "detray/test/utils/types.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

using namespace detray;

using algebra_t = test::algebra;
using matrix_operator = test::matrix_operator;

// Test is done for muon
namespace {
pdg_particle ptc = muon<scalar>();
}

// Material interaction test with telescope Geometry
GTEST_TEST(detray_material, telescope_geometry_energy_loss) {

    vecmem::host_memory_resource host_mr;

    // Build in x-direction from given module positions
    detail::ray<algebra_t> traj{{0.f, 0.f, 0.f}, 0.f, {1.f, 0.f, 0.f}, -1.f};
    std::vector<scalar> positions = {0.f,   50.f,  100.f, 150.f, 200.f, 250.f,
                                     300.f, 350.f, 400.f, 450.f, 500.f};

    const auto mat = silicon_tml<scalar>();
    constexpr scalar thickness{0.17f * unit<scalar>::cm};

    tel_det_config<rectangle2D> tel_cfg{20.f * unit<scalar>::mm,
                                        20.f * unit<scalar>::mm};
    tel_cfg.positions(positions)
        .pilot_track(traj)
        .module_material(mat)
        .mat_thickness(thickness);

    const auto [det, names] = build_telescope_detector(host_mr, tel_cfg);

    using navigator_t = navigator<decltype(det)>;
    using stepper_t = line_stepper<algebra_t>;
    using interactor_t = pointwise_material_interactor<algebra_t>;
    using actor_chain_t =
        actor_chain<dtuple, pathlimit_aborter, parameter_transporter<algebra_t>,
                    interactor_t, parameter_resetter<algebra_t>>;
    using propagator_t = propagator<stepper_t, navigator_t, actor_chain_t>;

    // Propagator is built from the stepper and navigator
    propagation::config prop_cfg{};
    prop_cfg.navigation.overstep_tolerance = -100.f * unit<float>::um;
    propagator_t p{prop_cfg};

    constexpr scalar q{-1.f};
    constexpr scalar iniP{10.f * unit<scalar>::GeV};

    // Bound vector
    bound_parameters_vector<algebra_t> bound_vector{};
    bound_vector.set_theta(constant<scalar>::pi_2);
    bound_vector.set_qop(q / iniP);

    typename bound_track_parameters<algebra_t>::covariance_type bound_cov =
        matrix_operator().template zero<e_bound_size, e_bound_size>();

    // bound track parameter at first physical plane
    const bound_track_parameters<algebra_t> bound_param(
        geometry::barcode{}.set_index(0u), bound_vector, bound_cov);

    pathlimit_aborter::state aborter_state{};
    parameter_transporter<algebra_t>::state bound_updater{};
    interactor_t::state interactor_state{};
    parameter_resetter<algebra_t>::state parameter_resetter_state{};

    // Create actor states tuples
    auto actor_states = detray::tie(aborter_state, bound_updater,
                                    interactor_state, parameter_resetter_state);

    propagator_t::state state(bound_param, det);
    state.do_debug = true;

    // Propagate the entire detector
    ASSERT_TRUE(p.propagate(state, actor_states))
        << state.debug_stream.str() << std::endl;

    // new momentum
    const scalar newP{state._stepping.bound_params().p(ptc.charge())};

    // mass
    const auto mass = ptc.mass();

    // new energy
    const scalar newE{std::hypot(newP, mass)};

    // Initial energy
    const scalar iniE{std::hypot(iniP, mass)};

    // New qop variance
    const scalar new_var_qop{
        matrix_operator().element(state._stepping.bound_params().covariance(),
                                  e_bound_qoverp, e_bound_qoverp)};

    // Interaction object
    interaction<scalar> I;

    // Zero incidence angle
    const scalar cos_inc_ang{1.f};

    // Same material used for default telescope detector
    material_slab<scalar> slab(mat, thickness);

    // Path segment in the material
    const scalar path_segment{slab.path_segment(cos_inc_ang)};

    // Expected Bethe Stopping power for telescope geometry is estimated
    // as (number of planes * energy loss per plane assuming 1 GeV muon).
    // It is not perfectly precise as the track loses its energy during
    // propagation. However, since the energy loss << the track momentum,
    // the assumption is not very bad
    const scalar dE{
        I.compute_energy_loss_bethe_bloch(path_segment, slab.get_material(),
                                          ptc, {ptc, ptc.charge() / iniP}) *
        static_cast<scalar>(positions.size())};

    // Check if the new energy after propagation is enough close to the
    // expected value
    EXPECT_NEAR(newE, iniE - dE, 1e-5f);

    const scalar sigma_qop{I.compute_energy_loss_landau_sigma_QOverP(
        path_segment, slab.get_material(), ptc, {ptc, ptc.charge() / iniP})};

    const scalar dvar_qop{sigma_qop * sigma_qop *
                          static_cast<scalar>(positions.size() - 1u)};

    EXPECT_NEAR(new_var_qop, dvar_qop, 1e-10f);

    // @todo: Validate the backward direction case as well?
}

// Material interaction test with telescope Geometry
GTEST_TEST(detray_material, telescope_geometry_scattering_angle) {
    vecmem::host_memory_resource host_mr;

    // Build in x-direction from given module positions
    detail::ray<algebra_t> traj{{0.f, 0.f, 0.f}, 0.f, {1.f, 0.f, 0.f}, -1.f};
    std::vector<scalar> positions = {0.f};

    // To make sure that someone won't put more planes than one by accident
    EXPECT_EQ(positions.size(), 1u);

    // Material
    const auto mat = silicon_tml<scalar>();
    const scalar thickness = 100.f * unit<scalar>::cm;

    // Create telescope geometry
    tel_det_config<rectangle2D> tel_cfg{2000.f * unit<scalar>::mm,
                                        2000.f * unit<scalar>::mm};
    tel_cfg.positions(positions)
        .pilot_track(traj)
        .module_material(mat)
        .mat_thickness(thickness);

    const auto [det, names] = build_telescope_detector(host_mr, tel_cfg);

    using navigator_t = navigator<decltype(det)>;
    using stepper_t = line_stepper<algebra_t>;
    using simulator_t = random_scatterer<algebra_t>;
    using actor_chain_t =
        actor_chain<dtuple, pathlimit_aborter, parameter_transporter<algebra_t>,
                    simulator_t, parameter_resetter<algebra_t>>;
    using propagator_t = propagator<stepper_t, navigator_t, actor_chain_t>;

    // Propagator is built from the stepper and navigator
    propagation::config prop_cfg{};
    prop_cfg.navigation.overstep_tolerance = -100.f * unit<float>::um;
    propagator_t p{prop_cfg};

    constexpr scalar q{-1.f};
    constexpr scalar iniP{10.f * unit<scalar>::GeV};

    // Initial track parameters directing x-axis
    bound_parameters_vector<algebra_t> bound_vector{};
    bound_vector.set_theta(constant<scalar>::pi_2);
    bound_vector.set_qop(q / iniP);

    typename bound_track_parameters<algebra_t>::covariance_type bound_cov =
        matrix_operator().template zero<e_bound_size, e_bound_size>();

    // bound track parameter
    const bound_track_parameters<algebra_t> bound_param(
        geometry::barcode{}.set_index(0u), bound_vector, bound_cov);

    std::size_t n_samples{100000u};
    std::vector<scalar> phis;
    std::vector<scalar> thetas;

    for (std::size_t i = 0u; i < n_samples; i++) {

        pathlimit_aborter::state aborter_state{};
        parameter_transporter<algebra_t>::state bound_updater{};
        // Seed = sample id
        simulator_t::state simulator_state{i};
        simulator_state.do_energy_loss = false;
        parameter_resetter<algebra_t>::state parameter_resetter_state{};

        // Create actor states tuples
        auto actor_states =
            detray::tie(aborter_state, bound_updater, simulator_state,
                        parameter_resetter_state);

        propagator_t::state state(bound_param, det);
        state.do_debug = true;

        // Propagate the entire detector
        ASSERT_TRUE(p.propagate(state, actor_states))
            << state.debug_stream.str() << std::endl;

        const auto& final_param = state._stepping.bound_params();

        // Updated phi and theta variance
        if (i == 0u) {
            pointwise_material_interactor<algebra_t>{}.update_angle_variance(
                bound_cov, traj.dir(),
                simulator_state.projected_scattering_angle, 1);
        }

        phis.push_back(final_param.phi());
        thetas.push_back(final_param.theta());
    }
    scalar phi_variance{statistics::rms(phis, bound_param.phi())};
    scalar theta_variance{statistics::rms(thetas, bound_param.theta())};

    // Get the phi and theta variance
    scalar ref_phi_variance =
        getter::element(bound_cov, e_bound_phi, e_bound_phi);
    scalar ref_theta_variance =
        getter::element(bound_cov, e_bound_theta, e_bound_theta);

    // Tolerate upto 1% difference
    EXPECT_NEAR((phi_variance - ref_phi_variance) / ref_phi_variance, 0.f,
                1e-2f);
    EXPECT_NEAR((theta_variance - ref_theta_variance) / ref_theta_variance, 0.f,
                1e-2f);

    // To make sure that the variances are not zero
    EXPECT_TRUE(ref_phi_variance > 1e-9f && ref_theta_variance > 1e-9f);
}

// Material interaction test with telescope Geometry with volume material
GTEST_TEST(detray_material, telescope_geometry_volume_material) {

    vecmem::host_memory_resource host_mr;

    // Propagator types
    using bfield_t = bfield::const_field_t;
    using stepper_t = rk_stepper<bfield_t::view_t, algebra_t>;
    using actor_chain_t = actor_chain<dtuple, pathlimit_aborter>;
    using vector3 = test::vector3;

    // Bfield setup
    vector3 B_z{0.f, 0.f, 2.f * unit<scalar>::T};
    const bfield_t const_bfield = bfield::create_const_field(B_z);

    // Track setup
    constexpr scalar q{-1.f};
    constexpr scalar iniP{10.f * unit<scalar>::GeV};

    bound_parameters_vector<algebra_t> bound_vector{};
    bound_vector.set_theta(constant<scalar>::pi_2);
    bound_vector.set_qop(q / iniP);

    typename bound_track_parameters<algebra_t>::covariance_type bound_cov =
        matrix_operator().template zero<e_bound_size, e_bound_size>();

    // bound track parameter at first physical plane
    const bound_track_parameters<algebra_t> bound_param(
        geometry::barcode{}.set_index(0u), bound_vector, bound_cov);

    // Create actor states tuples
    const scalar path_limit = 100 * unit<scalar>::mm;

    // Build in x-direction from given module positions
    detail::ray<algebra_t> traj{{0.f, 0.f, 0.f}, 0.f, {1.f, 0.f, 0.f}, -1.f};
    std::vector<scalar> positions = {0.f, 10000.f * unit<scalar>::mm};

    // NO material at modules
    const auto module_mat = vacuum<scalar>();

    // Create telescope geometry
    tel_det_config<rectangle2D> tel_cfg{100000.f * unit<scalar>::mm,
                                        100000.f * unit<scalar>::mm};
    tel_cfg.positions(positions).pilot_track(traj).module_material(module_mat);

    std::vector<material<scalar>> vol_mats = {
        vacuum<scalar>(), isobutane<scalar>(), silicon<scalar>(),
        tungsten<scalar>()};

    for (const auto& mat : vol_mats) {
        tel_cfg.volume_material(mat);
        const auto [det, names] = build_telescope_detector(host_mr, tel_cfg);

        using navigator_t = navigator<decltype(det)>;
        using propagator_t = propagator<stepper_t, navigator_t, actor_chain_t>;

        // Propagator is built from the stepper and navigator
        propagation::config prop_cfg{};
        prop_cfg.navigation.overstep_tolerance = -100.f * unit<float>::um;
        propagator_t p{prop_cfg};

        propagator_t::state state(bound_param, const_bfield, det);

        pathlimit_aborter::state abrt_state{path_limit};
        auto actor_states = detray::tie(abrt_state);

        p.propagate(state, actor_states);

        const auto newP = state._stepping().p(ptc.charge());
        const auto mass = ptc.mass();

        const auto eloss_approx =
            interaction<scalar>().compute_energy_loss_bethe_bloch(
                state._stepping.path_length(), mat, ptc,
                {ptc, bound_param.qop()});

        const auto iniE = std::sqrt(iniP * iniP + mass * mass);
        const auto newE = std::sqrt(newP * newP + mass * mass);
        const auto eloss = iniE - newE;

        if (mat == vacuum<scalar>()) {
            ASSERT_FLOAT_EQ(float(eloss), 0.f);
        } else {
            ASSERT_TRUE(eloss > 0.f);
        }

        ASSERT_NEAR(eloss, eloss_approx, eloss * 0.01);
    }
}
