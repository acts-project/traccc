/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/definitions/units.hpp"

// Detray test include(s)
#include "detray/test/common/detail/register_checks.hpp"
#include "detray/test/common/detail/whiteboard.hpp"
#include "detray/test/cpu/detector_consistency.hpp"
#include "detray/test/cpu/detector_scan.hpp"
#include "detray/test/cpu/material_scan.hpp"
#include "detray/test/cpu/material_validation.hpp"
#include "detray/test/cpu/navigation_validation.hpp"
#include "detray/test/utils/detectors/build_toy_detector.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s)
#include <gtest/gtest.h>

int main(int argc, char **argv) {

    using namespace detray;

    // Filter out the google test flags
    ::testing::InitGoogleTest(&argc, argv);

    using toy_detector_t = detector<toy_metadata>;
    using scalar_t = typename toy_detector_t::scalar_type;

    //
    // Toy detector configuration
    //
    toy_det_config toy_cfg{};
    toy_cfg.n_brl_layers(4u).n_edc_layers(7u);
    toy_cfg.use_material_maps(true);

    std::cout << toy_cfg << std::endl;

    // Build the geometry
    vecmem::host_memory_resource host_mr;
    auto [toy_det, toy_names] = build_toy_detector(host_mr, toy_cfg);

    auto white_board = std::make_shared<test::whiteboard>();

    // General data consistency of the detector
    test::consistency_check<toy_detector_t>::config cfg_cons{};
    detail::register_checks<test::consistency_check>(
        toy_det, toy_names, cfg_cons.name("toy_detector_consistency"));

    // Navigation link consistency, discovered by ray intersection
    test::ray_scan<toy_detector_t>::config cfg_ray_scan{};
    cfg_ray_scan.name("toy_detector_ray_scan");
    cfg_ray_scan.whiteboard(white_board);
    cfg_ray_scan.track_generator().n_tracks(10000u);

    detail::register_checks<test::ray_scan>(toy_det, toy_names, cfg_ray_scan);

    // Comparison of straight line navigation with ray scan
    test::straight_line_navigation<toy_detector_t>::config cfg_str_nav{};
    cfg_str_nav.name("toy_detector_straight_line_navigation");
    cfg_str_nav.whiteboard(white_board);
    cfg_str_nav.propagation().navigation.search_window = {3u, 3u};
    auto mask_tolerance = cfg_ray_scan.mask_tolerance();
    cfg_str_nav.propagation().navigation.min_mask_tolerance =
        static_cast<float>(mask_tolerance[0]);
    cfg_str_nav.propagation().navigation.max_mask_tolerance =
        static_cast<float>(mask_tolerance[1]);

    detail::register_checks<test::straight_line_navigation>(toy_det, toy_names,
                                                            cfg_str_nav);

    // Navigation link consistency, discovered by helix intersection
    test::helix_scan<toy_detector_t>::config cfg_hel_scan{};
    cfg_hel_scan.name("toy_detector_helix_scan");
    cfg_hel_scan.whiteboard(white_board);
    // Let the Newton algorithm dynamically choose tol. based on approx. error
    cfg_hel_scan.mask_tolerance({detray::detail::invalid_value<scalar_t>(),
                                 detray::detail::invalid_value<scalar_t>()});
    cfg_hel_scan.track_generator().n_tracks(10000u);
    cfg_hel_scan.track_generator().randomize_charge(true);
    cfg_hel_scan.track_generator().eta_range(-4.f, 4.f);
    cfg_hel_scan.track_generator().p_T(1.f * unit<scalar_t>::GeV);

    detail::register_checks<test::helix_scan>(toy_det, toy_names, cfg_hel_scan);

    // Comparison of navigation in a constant B-field with helix
    test::helix_navigation<toy_detector_t>::config cfg_hel_nav{};
    cfg_hel_nav.name("toy_detector_helix_navigation");
    cfg_hel_nav.whiteboard(white_board);
    cfg_hel_nav.propagation().navigation.search_window = {3u, 3u};

    detail::register_checks<test::helix_navigation>(toy_det, toy_names,
                                                    cfg_hel_nav);

    // Run the material validation - Material Maps
    test::material_scan<toy_detector_t>::config mat_scan_cfg{};
    mat_scan_cfg.name("toy_detector_material_scan");
    mat_scan_cfg.whiteboard(white_board);
    mat_scan_cfg.track_generator().uniform_eta(true).eta_range(-4.f, 4.f);
    mat_scan_cfg.track_generator().phi_steps(100).eta_steps(100);

    // Record the material using a ray scan
    detail::register_checks<test::material_scan>(toy_det, toy_names,
                                                 mat_scan_cfg);

    // Now trace the material during navigation and compare
    test::material_validation<toy_detector_t>::config mat_val_cfg{};
    mat_val_cfg.name("toy_detector_material_validaiton");
    mat_val_cfg.whiteboard(white_board);
    // Reduce tolerance for single precision tests
    if constexpr (std::is_same_v<scalar_t, float>) {
        mat_val_cfg.relative_error(1.5e-6f);
    }
    mat_val_cfg.propagation() = cfg_str_nav.propagation();

    // @TODO: Put material maps on all portals
    detail::register_checks<test::material_validation>(toy_det, toy_names,
                                                       mat_val_cfg);

    // Run the material validation - Homogeneous material
    toy_cfg.use_material_maps(false);

    std::cout << toy_cfg << std::endl;

    auto [toy_det_hom_mat, toy_names_hom_mat] =
        build_toy_detector(host_mr, toy_cfg);
    toy_names_hom_mat.at(0) += "_hom_material";

    // Check that the detector was built correctly
    detail::register_checks<test::consistency_check>(
        toy_det_hom_mat, toy_names_hom_mat,
        cfg_cons.name("toy_detector_consistency_hom_mat"));

    // Record the material using a ray scan
    mat_scan_cfg.name("toy_detector_hom_material_scan");
    detail::register_checks<test::material_scan>(
        toy_det_hom_mat, toy_names_hom_mat, mat_scan_cfg);

    // Now trace the material during navigation and compare
    mat_val_cfg.name("toy_detector_hom_material_validaiton");
    detail::register_checks<test::material_validation>(
        toy_det_hom_mat, toy_names_hom_mat, mat_val_cfg);

    // Run the checks
    return RUN_ALL_TESTS();
}
