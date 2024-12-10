/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/detectors/bfield.hpp"

// Detray test include(s)
#include "detray/test/common/detail/register_checks.hpp"
#include "detray/test/common/detail/whiteboard.hpp"
#include "detray/test/cpu/detector_scan.hpp"
#include "detray/test/cpu/material_scan.hpp"
#include "detray/test/device/cuda/material_validation.hpp"
#include "detray/test/device/cuda/navigation_validation.hpp"
#include "detray/test/utils/detectors/build_telescope_detector.hpp"

// Vecmem include(s)
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// GTest include
#include <gtest/gtest.h>

// System include(s)
#include <limits>

using namespace detray;

int main(int argc, char **argv) {

    using namespace detray;

    // Filter out the google test flags
    ::testing::InitGoogleTest(&argc, argv);

    /// Vecmem memory resource for the device allocations
    vecmem::cuda::device_memory_resource dev_mr{};

    //
    // Telescope detector configuration
    //
    using tel_detector_t = detector<telescope_metadata<rectangle2D>>;
    using scalar_t = typename tel_detector_t::scalar_type;

    tel_det_config<rectangle2D> tel_cfg{20.f * unit<scalar_t>::mm,
                                        20.f * unit<scalar_t>::mm};
    tel_cfg.n_surfaces(10u)
        .length(500.f * unit<scalar_t>::mm)
        .envelope(500.f * unit<scalar_t>::um);

    vecmem::host_memory_resource host_mr;

    const auto [tel_det, tel_names] =
        build_telescope_detector(host_mr, tel_cfg);

    auto white_board = std::make_shared<test::whiteboard>();

    // Navigation link consistency, discovered by ray intersection
    test::ray_scan<tel_detector_t>::config cfg_ray_scan{};
    cfg_ray_scan.name("telescope_detector_ray_scan_for_cuda");
    cfg_ray_scan.whiteboard(white_board);
    cfg_ray_scan.track_generator().n_tracks(1000u);
    // The first surface is at z=0, so shift the track origin back
    cfg_ray_scan.track_generator().origin({0.f, 0.f, -0.05f});
    cfg_ray_scan.track_generator().theta_range(
        0.f, 0.25f * constant<scalar_t>::pi_4);

    detail::register_checks<test::ray_scan>(tel_det, tel_names, cfg_ray_scan);

    // Comparison of straight line navigation with ray scan
    detray::cuda::straight_line_navigation<tel_detector_t>::config
        cfg_str_nav{};
    cfg_str_nav.name("telescope_detector_straight_line_navigation_cuda");
    cfg_str_nav.whiteboard(white_board);
    auto mask_tolerance = cfg_ray_scan.mask_tolerance();
    cfg_str_nav.propagation().navigation.min_mask_tolerance =
        static_cast<float>(mask_tolerance[0]);
    cfg_str_nav.propagation().navigation.max_mask_tolerance =
        static_cast<float>(mask_tolerance[1]);

    detail::register_checks<detray::cuda::straight_line_navigation>(
        tel_det, tel_names, cfg_str_nav);

    // Navigation link consistency, discovered by helix intersection
    test::helix_scan<tel_detector_t>::config cfg_hel_scan{};
    cfg_hel_scan.name("telescope_detector_helix_scan_for_cuda");
    cfg_hel_scan.whiteboard(white_board);
    // Let the Newton algorithm dynamically choose tol. based on approx. error
    cfg_hel_scan.mask_tolerance({detray::detail::invalid_value<scalar_t>(),
                                 detray::detail::invalid_value<scalar_t>()});
    cfg_hel_scan.track_generator().n_tracks(1000u);
    cfg_hel_scan.track_generator().p_tot(10.f * unit<scalar_t>::GeV);
    cfg_hel_scan.track_generator().origin({0.f, 0.f, -0.05f});
    cfg_hel_scan.track_generator().theta_range(
        0.f, 0.25f * constant<scalar_t>::pi_4);

    detail::register_checks<test::helix_scan>(tel_det, tel_names, cfg_hel_scan);

    // Comparison of navigation in a constant B-field with helix
    detray::cuda::helix_navigation<tel_detector_t>::config cfg_hel_nav{};
    cfg_hel_nav.name("telescope_detector_helix_navigation_cuda");
    cfg_hel_nav.whiteboard(white_board);
    cfg_hel_nav.propagation().navigation.overstep_tolerance =
        -100.f * unit<float>::um;

    detail::register_checks<detray::cuda::helix_navigation>(tel_det, tel_names,
                                                            cfg_hel_nav);

    // Run the material validation
    test::material_scan<tel_detector_t>::config mat_scan_cfg{};
    mat_scan_cfg.name("telescope_detector_material_scan_for_cuda");
    mat_scan_cfg.whiteboard(white_board);
    mat_scan_cfg.track_generator().uniform_eta(true).eta_range(1.f, 6.f);
    mat_scan_cfg.track_generator().origin({0.f, 0.f, -0.05f});
    mat_scan_cfg.track_generator().phi_steps(10).eta_steps(100);

    // Record the material using a ray scan
    detail::register_checks<test::material_scan>(tel_det, tel_names,
                                                 mat_scan_cfg);

    // Now trace the material during navigation and compare
    detray::cuda::material_validation<tel_detector_t>::config mat_val_cfg{};
    mat_val_cfg.name("telescope_detector_material_validaiton_cuda");
    mat_val_cfg.whiteboard(white_board);
    mat_val_cfg.device_mr(&dev_mr);
    mat_val_cfg.propagation() = cfg_str_nav.propagation();

    detail::register_checks<detray::cuda::material_validation>(
        tel_det, tel_names, mat_val_cfg);

    // Run the checks
    return RUN_ALL_TESTS();
}
