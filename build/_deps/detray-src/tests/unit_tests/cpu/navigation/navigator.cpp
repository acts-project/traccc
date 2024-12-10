/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/navigation/navigator.hpp"

#include "detray/definitions/detail/indexing.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/line_stepper.hpp"
#include "detray/tracks/tracks.hpp"

// Detray test include(s)
#include "detray/test/utils/detectors/build_toy_detector.hpp"
#include "detray/test/utils/detectors/build_wire_chamber.hpp"
#include "detray/test/utils/inspectors.hpp"

// Test include(s)
#include "detray/test/utils/types.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GoogleTest include(s)
#include <gtest/gtest.h>

// System include(s)
#include <map>

namespace detray {

namespace {

constexpr std::size_t cache_size{navigation::default_cache_size};

// dummy propagator state
template <typename stepping_t, typename navigation_t>
struct prop_state {
    using context_t = typename navigation_t::detector_type::geometry_context;
    stepping_t _stepping;
    navigation_t _navigation;
    context_t _context{};
};

/// Checks for a correct 'towards_surface' state
template <typename navigator_t, typename state_t = typename navigator_t::state>
inline void check_towards_surface(state_t &state, dindex vol_id,
                                  std::size_t n_candidates, dindex next_id) {
    ASSERT_EQ(state.status(), navigation::status::e_towards_object);
    ASSERT_EQ(state.volume(), vol_id);
    ASSERT_EQ(state.n_candidates(), std::min(n_candidates, cache_size - 1));
    // the portal is still the next object, since we did not step
    ASSERT_EQ(state.next_surface().index(), next_id);
    ASSERT_TRUE((state.trust_level() == navigation::trust_level::e_full) ||
                (state.trust_level() == navigation::trust_level::e_high));
}

/// Checks for a correct 'on_surface' state
template <typename navigator_t, typename state_t = typename navigator_t::state>
inline void check_on_surface(state_t &state, dindex vol_id,
                             std::size_t n_candidates, dindex current_id,
                             dindex next_id) {
    // The status is: on surface/towards surface if the next candidate is
    // immediately updated and set in the same update call
    ASSERT_TRUE(state.status() == navigation::status::e_on_module ||
                state.status() == navigation::status::e_on_portal);
    // Points towards next candidate
    ASSERT_TRUE(std::abs(state()) >= 1.f * unit<scalar>::um);
    ASSERT_EQ(state.volume(), vol_id);
    ASSERT_EQ(state.n_candidates(), std::min(n_candidates, cache_size - 1));
    ASSERT_EQ(state.barcode().volume(), vol_id);
    ASSERT_EQ(state.barcode().index(), current_id);
    // points to the next surface now
    ASSERT_EQ(state.next_surface().index(), next_id);
    ASSERT_EQ(state.trust_level(), navigation::trust_level::e_full);
}

/// Checks for a correctly handled volume switch
template <typename navigator_t, typename state_t = typename navigator_t::state>
inline void check_volume_switch(state_t &state, dindex vol_id) {
    // Switched to next volume
    ASSERT_EQ(state.volume(), vol_id);
    // The status is towards first surface in new volume
    ASSERT_EQ(state.status(), navigation::status::e_on_portal);
    // Kernel is newly initialized
    ASSERT_FALSE(state.is_exhausted());
    ASSERT_EQ(state.trust_level(), navigation::trust_level::e_full);
}

/// Checks an entire step onto the next surface
template <typename navigator_t, typename stepper_t, typename prop_state_t>
inline void step_and_check(navigator_t &nav, stepper_t &stepper,
                           prop_state_t &propagation,
                           const navigation::config &nav_cfg,
                           const stepping::config &step_cfg, dindex vol_id,
                           std::size_t n_candidates, dindex current_id,
                           dindex next_id) {
    auto &navigation = propagation._navigation;
    auto &stepping = propagation._stepping;

    // Step onto the surface in volume
    stepper.step(navigation(), stepping, step_cfg);
    navigation.set_high_trust();
    // Stepper reduced trust level
    ASSERT_TRUE(navigation.trust_level() == navigation::trust_level::e_high);
    nav.update(stepping(), navigation, nav_cfg);
    ASSERT_TRUE(navigation.is_alive());
    // Trust level is restored
    ASSERT_EQ(navigation.trust_level(), navigation::trust_level::e_full);
    // The status is on surface
    check_on_surface<navigator_t>(navigation, vol_id, n_candidates, current_id,
                                  next_id);
}

}  // anonymous namespace

}  // namespace detray

/// This tests the construction and general methods of the navigator
GTEST_TEST(detray_navigation, navigator_toy_geometry) {
    using namespace detray;
    using namespace detray::navigation;

    using algebra_t = test::algebra;
    using point3 = test::point3;
    using vector3 = test::vector3;

    vecmem::host_memory_resource host_mr;

    /// Tolerance for tests
    constexpr double tol{0.01};

    auto [toy_det, names] = build_toy_detector(host_mr);

    using detector_t = decltype(toy_det);
    using inspector_t = navigation::print_inspector;
    using navigator_t = navigator<detector_t, cache_size, inspector_t>;
    using constraint_t = constrained_step<>;
    using stepper_t = line_stepper<algebra_t, constraint_t>;

    // State type in the nominal navigation (no inspectors)
    using nav_stat_t = navigator<detector_t, cache_size>::state;

    // 256 bytes for single precision
    static_assert(sizeof(nav_stat_t) ==
                  16 + navigation::default_cache_size *
                           sizeof(typename nav_stat_t::value_type));

    // test track
    point3 pos{0.f, 0.f, 0.f};
    vector3 mom{1.f, 1.f, 0.f};
    free_track_parameters<algebra_t> traj(pos, 0.f, mom, -1.f);

    stepper_t stepper;
    navigator_t nav;
    navigation::config nav_cfg{};
    nav_cfg.search_window = {3u, 3u};

    stepping::config step_cfg{};

    prop_state<stepper_t::state, navigator_t::state> propagation{
        stepper_t::state{traj}, navigator_t::state(toy_det)};
    navigator_t::state &navigation = propagation._navigation;
    stepper_t::state &stepping = propagation._stepping;
    const auto &ctx = propagation._context;

    // Check that the state is unitialized
    // Default volume is zero
    ASSERT_EQ(navigation.volume(), 0u);
    // No surface candidates
    ASSERT_EQ(navigation.n_candidates(), 0u);
    // You can not trust the state
    ASSERT_EQ(navigation.trust_level(), trust_level::e_no_trust);
    // The status is unkown
    ASSERT_EQ(navigation.status(), status::e_unknown);

    //
    // beampipe
    //

    // Initialize navigation
    // Test that the navigator has a heartbeat
    nav.init(stepping(), navigation, nav_cfg, ctx);
    ASSERT_TRUE(navigation.is_alive());
    // The status is towards beampipe
    // Two candidates: beampipe and portal
    // First candidate is the beampipe
    check_towards_surface<navigator_t>(navigation, 0u, 2u, 0u);
    // Distance to beampipe surface
    ASSERT_NEAR(navigation(), 19.f, tol);

    // Let's make half the step towards the beampipe
    stepping.template set_constraint<step::constraint::e_user>(navigation() *
                                                               0.5f);
    stepper.step(navigation(), stepping, step_cfg);
    // Navigation policy might reduce trust level to fair trust
    navigation.set_fair_trust();
    // Release user constraint again
    stepping.template release_step<step::constraint::e_user>();
    ASSERT_TRUE(navigation.trust_level() == trust_level::e_fair);
    // Re-navigate
    nav.update(stepping(), navigation, nav_cfg);
    ASSERT_TRUE(navigation.is_alive());
    // Trust level is restored
    ASSERT_EQ(navigation.trust_level(), trust_level::e_full);
    // The status remains: towards surface
    check_towards_surface<navigator_t>(navigation, 0u, 2u, 0u);
    // Distance to beampipe is now halved
    ASSERT_NEAR(navigation(), 9.5f, tol);

    // Let's immediately update, nothing should change, as there is full trust
    nav.update(stepping(), navigation, nav_cfg);
    ASSERT_TRUE(navigation.is_alive());
    check_towards_surface<navigator_t>(navigation, 0u, 2u, 0u);
    ASSERT_NEAR(navigation(), 9.5f, tol);

    // Now step onto the beampipe (idx 0)
    step_and_check(nav, stepper, propagation, nav_cfg, step_cfg, 0u, 1u, 0u,
                   8u);
    // New target: Distance to the beampipe volume cylinder portal
    ASSERT_NEAR(navigation(), 6.f, tol);

    // Step onto portal 7 in volume 0
    stepper.step(navigation(), stepping, step_cfg);
    navigation.set_high_trust();
    ASSERT_TRUE(navigation.trust_level() == trust_level::e_high);
    nav.update(stepping(), navigation, nav_cfg);
    ASSERT_TRUE(navigation.is_alive()) << navigation.inspector().to_string();
    ASSERT_EQ(navigation.trust_level(), trust_level::e_full);
    ASSERT_EQ(navigation.volume(), 8u);

    //
    // barrel
    //

    // Last volume before we leave world
    dindex last_vol_id = 15u;

    // maps volume id to the sequence of surfaces that the navigator encounters
    std::vector<std::pair<dindex, std::vector<dindex>>> sf_sequences;

    // gap 1
    sf_sequences.emplace_back(8u, std::vector<dindex>{600u, 601u});
    // layer 1
    sf_sequences.emplace_back(
        7u, std::vector<dindex>{596u, 493u, 477u, 494u, 478u, 597u});
    // gap 2
    sf_sequences.emplace_back(10u, std::vector<dindex>{1056u, 1057u});
    // layer 2
    sf_sequences.emplace_back(
        9u, std::vector<dindex>{1052u, 847u, 815u, 848u, 816u, 1053u});
    // gap 3
    sf_sequences.emplace_back(12u, std::vector<dindex>{1792u, 1793u});
    // layer 3
    sf_sequences.emplace_back(11u,
                              std::vector<dindex>{1788u, 1456u, 1404u, 1789u});
    // gap 4
    sf_sequences.emplace_back(14u, std::vector<dindex>{2892u, 2893u});
    // layer 4
    sf_sequences.emplace_back(13u,
                              std::vector<dindex>{2888u, 2390u, 2312u, 2889u});
    // gap 5
    sf_sequences.emplace_back(last_vol_id, std::vector<dindex>{2896u, 2897u});

    // Every iteration steps through one barrel layer
    for (const auto &[vol_id, sf_seq] : sf_sequences) {
        // Exclude the portal we are already on
        std::size_t n_candidates = sf_seq.size() - 1u;

        // Test the copy constructor of the propagation state
        auto propagation_cpy{propagation};
        navigator_t::state &navigation_cpy = propagation_cpy._navigation;
        stepper_t::state &stepping_cpy = propagation_cpy._stepping;

        // We switched to next barrel volume
        check_volume_switch<navigator_t>(navigation_cpy, vol_id);

        // The status is: on adjacent portal in volume, towards next candidate
        check_on_surface<navigator_t>(navigation_cpy, vol_id, n_candidates,
                                      sf_seq[0], sf_seq[1]);

        // Step through the module surfaces
        for (std::size_t sf = 1u; sf < sf_seq.size() - 1u; ++sf) {
            // Count only the currently reachable candidates
            step_and_check(nav, stepper, propagation_cpy, nav_cfg, step_cfg,
                           vol_id, n_candidates - sf, sf_seq[sf],
                           sf_seq[sf + 1u]);
        }

        // Step onto the portal in volume
        stepper.step(navigation_cpy(), stepping_cpy, step_cfg);
        navigation_cpy.set_high_trust();

        // Check agianst last volume
        if (vol_id == last_vol_id) {
            nav.update(stepping_cpy(), navigation_cpy, nav_cfg);
            ASSERT_FALSE(navigation_cpy.is_alive());
            // The status is: exited
            ASSERT_EQ(navigation_cpy.status(), status::e_on_target);
            // Switch to next volume leads out of the detector world -> exit
            ASSERT_TRUE(
                detray::detail::is_invalid_value(navigation_cpy.volume()));
            // We know we went out of the detector
            ASSERT_EQ(navigation_cpy.trust_level(), trust_level::e_full);
        } else {
            nav.update(stepping_cpy(), navigation_cpy, nav_cfg);
            ASSERT_TRUE(navigation_cpy.is_alive());
        }

        // Update the propagation state with current step (test assignment op)
        propagation = propagation_cpy;
    }

    // Leave for debugging
    // std::cout << navigation.inspector().to_string() << std::endl;
    ASSERT_TRUE(navigation.is_complete()) << navigation.inspector().to_string();
}

GTEST_TEST(detray_navigation, navigator_wire_chamber) {

    using namespace detray;
    using namespace detray::navigation;

    using algebra_t = test::algebra;
    using point3 = test::point3;
    using vector3 = test::vector3;

    vecmem::host_memory_resource host_mr;

    /// Tolerance for tests
    constexpr double tol{0.01};

    constexpr std::size_t n_layers{10};
    wire_chamber_config<> wire_cfg{};
    auto [wire_det, names] = build_wire_chamber(host_mr, wire_cfg);

    using detector_t = decltype(wire_det);
    using inspector_t = navigation::print_inspector;
    using navigator_t = navigator<detector_t, cache_size, inspector_t>;
    using constraint_t = constrained_step<>;
    using stepper_t = line_stepper<algebra_t, constraint_t>;

    // test track
    point3 pos{0.f, 0.f, 0.f};
    vector3 mom{0.f, 1.f, 0.f};
    free_track_parameters<algebra_t> traj(pos, 0.f, mom, -1.f);

    stepper_t stepper;
    navigator_t nav;
    navigation::config nav_cfg{};
    nav_cfg.mask_tolerance_scalor = 1e-2f;
    nav_cfg.path_tolerance = 1.f * unit<float>::um;
    nav_cfg.search_window = {3u, 3u};

    stepping::config step_cfg{};

    prop_state<stepper_t::state, navigator_t::state> propagation{
        stepper_t::state{traj}, navigator_t::state(wire_det)};
    navigator_t::state &navigation = propagation._navigation;
    stepper_t::state &stepping = propagation._stepping;
    const auto &ctx = propagation._context;

    // Check that the state is unitialized
    // Default volume is zero
    ASSERT_EQ(navigation.volume(), 0u);
    // No surface candidates
    ASSERT_EQ(navigation.n_candidates(), 0u);
    // You can not trust the state
    ASSERT_EQ(navigation.trust_level(), trust_level::e_no_trust);
    // The status is unkown
    ASSERT_EQ(navigation.status(), status::e_unknown);

    //
    // Beam Collison region
    //

    // Initialize navigation
    // Test that the navigator has a heartbeat
    nav.init(stepping(), navigation, nav_cfg, ctx);
    ASSERT_TRUE(navigation.is_alive());
    // The status is towards portal
    // One candidates: barrel cylinder portal
    check_towards_surface<navigator_t>(navigation, 0u, 1u, 0u);
    // Distance to portal
    ASSERT_NEAR(navigation(), 500.f * unit<scalar>::mm, tol);

    // Let's make half the step towards the portal
    stepping.template set_constraint<step::constraint::e_user>(navigation() *
                                                               0.5f);
    stepper.step(navigation(), stepping, step_cfg);
    // Navigation policy might reduce trust level to fair trust
    navigation.set_fair_trust();
    // Release user constraint again
    stepping.template release_step<step::constraint::e_user>();
    ASSERT_TRUE(navigation.trust_level() == trust_level::e_fair);
    // Re-navigate
    nav.update(stepping(), navigation, nav_cfg);
    ASSERT_TRUE(navigation.is_alive());
    // Trust level is restored
    ASSERT_EQ(navigation.trust_level(), trust_level::e_full);
    // The status remains: towards surface
    check_towards_surface<navigator_t>(navigation, 0u, 1u, 0u);
    // Distance to portal is now halved
    ASSERT_NEAR(navigation(), 250.f * unit<scalar>::mm, tol);

    // Let's immediately update, nothing should change, as there is full trust
    nav.update(stepping(), navigation, nav_cfg);
    ASSERT_TRUE(navigation.is_alive());
    check_towards_surface<navigator_t>(navigation, 0u, 1u, 0u);
    ASSERT_NEAR(navigation(), 250.f * unit<scalar>::mm, tol);

    // Step onto portal in volume 0
    stepper.step(navigation(), stepping, step_cfg);
    navigation.set_high_trust();
    ASSERT_TRUE(navigation.trust_level() == trust_level::e_high);
    nav.update(stepping(), navigation, nav_cfg);
    ASSERT_TRUE(navigation.is_alive()) << navigation.inspector().to_string();
    ASSERT_EQ(navigation.trust_level(), trust_level::e_full);

    //
    // Wire Layer
    //

    // Last volume before we leave world
    dindex last_vol_id = n_layers;

    // maps volume id to the sequence of surfaces that the navigator encounters
    std::map<dindex, std::vector<dindex>> sf_sequences;

    // layer 1 to 10
    sf_sequences[1] = {3u, 47u, 4u};
    sf_sequences[2] = {168u, 214u, 169u};
    sf_sequences[3] = {339u, 386u, 340u};
    sf_sequences[4] = {516u, 565u, 517u};
    sf_sequences[5] = {700u, 750u, 701u};
    sf_sequences[6] = {890u, 942u, 891u};
    sf_sequences[7] = {1086u, 1139u, 1087u};
    sf_sequences[8] = {1288u, 1343u, 1289u};
    sf_sequences[9] = {1497u, 1554u, 1498u};
    sf_sequences[10] = {1712u, 1770u, 1713u};

    // Every iteration steps through one wire layer
    for (const auto &[vol_id, sf_seq] : sf_sequences) {

        // Exclude the portal we are already on
        std::size_t n_candidates = sf_seq.size() - 1u;

        // Test the copy constructor of the propagation state
        auto propagation_cpy{propagation};
        navigator_t::state &navigation_cpy = propagation_cpy._navigation;
        stepper_t::state &stepping_cpy = propagation_cpy._stepping;

        // We switched to next barrel volume
        check_volume_switch<navigator_t>(navigation_cpy, vol_id);

        // The status is: on adjacent portal in volume, towards next candidate
        check_on_surface<navigator_t>(navigation_cpy, vol_id, n_candidates,
                                      sf_seq[0], sf_seq[1]);

        // Step through the module surfaces
        for (std::size_t sf = 1u; sf < sf_seq.size() - 1u; ++sf) {
            // Count only the currently reachable candidates
            step_and_check(nav, stepper, propagation_cpy, nav_cfg, step_cfg,
                           vol_id, n_candidates - sf, sf_seq[sf],
                           sf_seq[sf + 1u]);
        }

        // Step onto the portal in volume
        stepper.step(navigation_cpy(), stepping_cpy, step_cfg);
        navigation_cpy.set_high_trust();

        // Check agianst last volume
        if (vol_id == last_vol_id) {
            nav.update(stepping_cpy(), navigation_cpy, nav_cfg);
            ASSERT_FALSE(navigation_cpy.is_alive());
            // The status is: exited
            ASSERT_EQ(navigation_cpy.status(), status::e_on_target);
            // Switch to next volume leads out of the detector world -> exit
            ASSERT_TRUE(
                detray::detail::is_invalid_value(navigation_cpy.volume()));
            // We know we went out of the detector
            ASSERT_EQ(navigation_cpy.trust_level(), trust_level::e_full);
        } else {
            nav.update(stepping_cpy(), navigation_cpy, nav_cfg);
            ASSERT_TRUE(navigation_cpy.is_alive())
                << navigation_cpy.inspector().to_string();
        }

        // Update the propagation state with current step (test assignment op)
        propagation = propagation_cpy;
    }

    // Leave for debugging
    // std::cout << navigation.inspector().to_string() << std::endl;
    ASSERT_TRUE(navigation.is_complete()) << navigation.inspector().to_string();
}
