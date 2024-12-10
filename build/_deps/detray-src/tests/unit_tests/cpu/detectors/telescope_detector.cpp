/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/units.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes/unbounded.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/line_stepper.hpp"
#include "detray/propagator/propagation_config.hpp"
#include "detray/propagator/rk_stepper.hpp"
#include "detray/tracks/tracks.hpp"
#include "detray/utils/consistency_checker.hpp"

// Detray test include(s)
#include "detray/test/utils/detectors/build_telescope_detector.hpp"
#include "detray/test/utils/inspectors.hpp"
#include "detray/test/utils/types.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// Covfie include(s)
#include <covfie/core/backend/primitive/constant.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/vector.hpp>

// GTest include
#include <gtest/gtest.h>

// System include(s)
#include <utility>

namespace detray {

namespace {

using vector3 = test::vector3;
using algebra_t = test::algebra;

// dummy propagator state
template <typename stepping_t, typename navigation_t>
struct prop_state {
    using context_t = typename navigation_t::detector_type::geometry_context;

    stepping_t _stepping;
    navigation_t _navigation;
    context_t _context;

    template <typename track_t, typename field_type>
    prop_state(const track_t &t_in, const field_type &field,
               const typename navigation_t::detector_type &det,
               const context_t &ctx = {})
        : _stepping(t_in, field), _navigation(det), _context(ctx) {}
};

inline constexpr bool verbose_check = true;

const propagation::config prop_cfg{};

}  // anonymous namespace

}  // namespace detray

// This tests the construction and general methods of the navigator
GTEST_TEST(detray_detectors, telescope_detector) {

    using namespace detray;

    // Use rectangle surfaces
    mask<rectangle2D> rectangle{0u, 20.f * unit<scalar>::mm,
                                20.f * unit<scalar>::mm};
    tel_det_config<> tel_cfg{rectangle};

    using const_bfield_bknd_t =
        covfie::backend::constant<covfie::vector::vector_d<scalar, 3>,
                                  covfie::vector::vector_d<scalar, 3>>;
    using b_field_t = covfie::field<const_bfield_bknd_t>;

    using rk_stepper_t = rk_stepper<b_field_t::view_t, algebra_t>;
    using inspector_t = navigation::print_inspector;
    constexpr std::size_t cache_size{navigation::default_cache_size};

    // Test tolerance
    constexpr scalar tol{1e-4f};

    vecmem::host_memory_resource host_mr;

    // B-fields
    vector3 B_z{0.f, 0.f, 1.f * unit<scalar>::T};
    vector3 B_x{1.f * unit<scalar>::T, 0.f, 0.f};
    b_field_t b_field_z{covfie::make_parameter_pack(
        b_field_t::backend_t::configuration_t{B_z[0], B_z[1], B_z[2]})};
    b_field_t b_field_x{covfie::make_parameter_pack(
        b_field_t::backend_t::configuration_t{B_x[0], B_x[1], B_x[2]})};

    // steppers
    rk_stepper_t rk_stepper_z;
    rk_stepper_t rk_stepper_x;

    //
    // telescope along z
    //

    // Build from given module positions
    std::vector<scalar> positions = {0.f,   50.f,  100.f, 150.f, 200.f, 250.f,
                                     300.f, 350.f, 400.f, 450.f, 500.f};
    // Build telescope detector with unbounded planes
    const auto [z_tel_det1, z_tel_names1] =
        build_telescope_detector(host_mr, tel_cfg.positions(positions));

    // Some general checks
    const auto vol0 = tracking_volume{z_tel_det1, 0u};
    ASSERT_EQ(vol0.portals().size(), 6u);
    ASSERT_EQ(vol0.surfaces().size(), positions.size() + 6u);
    ASSERT_EQ(vol0.template surfaces<surface_id::e_sensitive>().size(),
              positions.size());

    // Test this only once, it is the same for all telescope detectors
    EXPECT_EQ(z_tel_names1.at(0u), "telescope_detector");
    EXPECT_EQ(z_tel_names1.at(1u), "telescope_world_0");

    // Check general consistency of the detector
    detail::check_consistency(z_tel_det1, verbose_check, z_tel_names1);

    // Build the same telescope detector with rectangular planes and given
    // length/number of surfaces
    tel_cfg.positions({}).n_surfaces(11u).length(500.f * unit<scalar>::mm);
    const auto [z_tel_det2, z_tel_names2] =
        build_telescope_detector(host_mr, tel_cfg);

    // Check general consistency of the detector
    detail::check_consistency(z_tel_det2, verbose_check, z_tel_names2);

    // Compare
    for (std::size_t i{0u}; i < z_tel_det1.surfaces().size(); ++i) {
        geometry::barcode bcd{};
        bcd.set_volume(0u).set_index(i);
        bcd.set_id((i == z_tel_det1.surfaces().size() - 1u)
                       ? surface_id::e_portal
                       : surface_id::e_sensitive);
        EXPECT_TRUE(z_tel_det1.surface(bcd) == z_tel_det2.surface(bcd));
    }

    //
    // telescope along x
    //

    // Same telescope, but in x direction and created from custom stepper
    detail::ray<algebra_t> x_track({0.f, 0.f, 0.f}, 0.f, {1.f, 0.f, 0.f}, -1.f);

    const auto [x_tel_det, x_tel_names] =
        build_telescope_detector(host_mr, tel_cfg.pilot_track(x_track));

    // Check general consistency of the detector
    detail::check_consistency(x_tel_det, verbose_check, x_tel_names);

    //
    // test propagation in all telescope detector instances
    //

    // Telescope navigation should be symmetric in x and z
    vector3 pos = {0.f, 0.f, 0.f};
    vector3 mom = {0.f, 0.f, 1.f};
    free_track_parameters<algebra_t> test_track_z1(pos, 0.f, mom, -1.f);
    free_track_parameters<algebra_t> test_track_z2(pos, 0.f, mom, -1.f);
    mom = {1.f, 0.f, 0.f};
    free_track_parameters<algebra_t> test_track_x(pos, 0.f, mom, -1.f);

    // navigators
    navigator<decltype(z_tel_det1), cache_size, inspector_t> navigator_z1;
    navigator<decltype(z_tel_det2), cache_size, inspector_t> navigator_z2;
    navigator<decltype(x_tel_det), cache_size, inspector_t> navigator_x;
    using navigation_state_t = decltype(navigator_z1)::state;
    using stepping_state_t = rk_stepper_t::state;

    // propagation states
    prop_state<stepping_state_t, navigation_state_t> propgation_z1(
        test_track_z1, b_field_z, z_tel_det1);
    prop_state<stepping_state_t, navigation_state_t> propgation_z2(
        test_track_z2, b_field_z, z_tel_det2);
    prop_state<stepping_state_t, navigation_state_t> propgation_x(
        test_track_x, b_field_x, x_tel_det);

    stepping_state_t &stepping_z1 = propgation_z1._stepping;
    stepping_state_t &stepping_z2 = propgation_z2._stepping;
    stepping_state_t &stepping_x = propgation_x._stepping;

    navigation_state_t &navigation_z1 = propgation_z1._navigation;
    navigation_state_t &navigation_z2 = propgation_z2._navigation;
    navigation_state_t &navigation_x = propgation_x._navigation;

    // propagate all telescopes
    navigator_z1.init(stepping_z1(), navigation_z1, prop_cfg.navigation,
                      prop_cfg.context);
    navigator_z2.init(stepping_z2(), navigation_z2, prop_cfg.navigation,
                      prop_cfg.context);
    navigator_x.init(stepping_x(), navigation_x, prop_cfg.navigation,
                     prop_cfg.context);

    bool heartbeat_z1 = navigation_z1.is_alive();
    bool heartbeat_z2 = navigation_z2.is_alive();
    bool heartbeat_x = navigation_x.is_alive();

    bool do_reset_z1{true};
    bool do_reset_z2{true};
    bool do_reset_x{true};

    while (heartbeat_z1 && heartbeat_z2 && heartbeat_x) {

        // check that all propagation flows are still running
        EXPECT_TRUE(heartbeat_z1);
        EXPECT_TRUE(heartbeat_z2);
        EXPECT_TRUE(heartbeat_x);

        heartbeat_z1 &= rk_stepper_z.step(navigation_z1(), stepping_z1,
                                          prop_cfg.stepping, do_reset_z1);
        heartbeat_z2 &= rk_stepper_z.step(navigation_z2(), stepping_z2,
                                          prop_cfg.stepping, do_reset_z2);
        heartbeat_x &= rk_stepper_x.step(navigation_x(), stepping_x,
                                         prop_cfg.stepping, do_reset_x);

        navigation_z1.set_high_trust();
        navigation_z2.set_high_trust();
        navigation_x.set_high_trust();

        do_reset_z1 = navigator_z1.update(stepping_z1(), navigation_z1,
                                          prop_cfg.navigation);
        do_reset_z2 = navigator_z2.update(stepping_z2(), navigation_z2,
                                          prop_cfg.navigation);
        do_reset_x =
            navigator_x.update(stepping_x(), navigation_x, prop_cfg.navigation);

        // Also reset when reached a surface
        do_reset_z1 |= navigation_z1.is_on_surface();
        do_reset_z2 |= navigation_z2.is_on_surface();
        do_reset_x |= navigation_x.is_on_surface();

        heartbeat_z1 &= navigation_z1.is_alive();
        heartbeat_z2 &= navigation_z2.is_alive();
        heartbeat_x &= navigation_x.is_alive();

        // The track path lengths should match between all propagations
        EXPECT_NEAR(
            std::abs(stepping_z1.path_length() - stepping_z2.path_length()) /
                stepping_z1.path_length(),
            0.f, tol);
        EXPECT_NEAR(
            std::abs(stepping_z1.path_length() - stepping_x.path_length()) /
                stepping_x.path_length(),
            0.f, tol);
        // The track positions in z should match exactly
        EXPECT_NEAR(getter::norm(stepping_z1().pos() - stepping_z2().pos()) /
                        getter::norm(stepping_z1().pos()),
                    0.f, tol);
        EXPECT_NEAR(getter::norm(stepping_z1().dir() - stepping_z2().dir()) /
                        getter::norm(stepping_z1().dir()),
                    0.f, tol);
    }

    // check that all propagation flows exited successfully
    ASSERT_TRUE(navigation_z1.is_complete())
        << navigation_z1.inspector().to_string();
    ASSERT_TRUE(navigation_z2.is_complete())
        << navigation_z2.inspector().to_string();
    ASSERT_TRUE(navigation_x.is_complete())
        << navigation_x.inspector().to_string();

    //
    // Build a telescope along a bent track
    //
    pos = {0.f, 0.f, 0.f};
    mom = {0.f, 1.f, 0.f};

    auto pilot_track = free_track_parameters<algebra_t>(pos, 0.f, mom, -1.f);

    detail::helix<algebra_t> helix_bz(pilot_track, &B_z);

    tel_det_config htel_cfg{rectangle, helix_bz};
    htel_cfg.n_surfaces(11u).length(500.f * unit<scalar>::mm);
    const auto [tel_detector, tel_names] =
        build_telescope_detector(host_mr, htel_cfg);

    // Check general consistency of the detector
    detail::check_consistency(tel_detector, verbose_check, tel_names);

    // make at least sure it is navigatable
    navigator<decltype(tel_detector), cache_size, inspector_t> tel_navigator;

    prop_state<stepping_state_t, navigation_state_t> tel_propagation(
        pilot_track, b_field_z, tel_detector);
    navigation_state_t &tel_navigation = tel_propagation._navigation;
    stepping_state_t &tel_stepping = tel_propagation._stepping;

    // run propagation
    tel_navigator.init(tel_stepping(), tel_navigation, prop_cfg.navigation,
                       prop_cfg.context);
    bool heartbeat_tel = tel_navigation.is_alive();

    bool do_reset_tel{true};

    while (heartbeat_tel) {
        heartbeat_tel &= rk_stepper_z.step(tel_navigation(), tel_stepping,
                                           prop_cfg.stepping, do_reset_tel);

        tel_navigation.set_high_trust();

        do_reset_tel = tel_navigator.update(tel_stepping(), tel_navigation,
                                            prop_cfg.navigation);

        do_reset_tel |= navigation_z1.is_on_surface();
        heartbeat_tel &= tel_navigation.is_alive();
    }
    // check that propagation was successful
    ASSERT_TRUE(tel_navigation.is_complete())
        << tel_navigation.inspector().to_string();
}
