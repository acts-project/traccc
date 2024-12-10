/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/navigation/detail/trajectories.hpp"

#include "detray/core/detector.hpp"
#include "detray/definitions/units.hpp"

// Detray plugin include(s)
#include "detray/plugins/svgtools/illustrator.hpp"
#include "detray/plugins/svgtools/writer.hpp"

// Detray test include(s)
#include "detray/test/utils/detectors/build_toy_detector.hpp"
#include "detray/test/validation/detector_scanner.hpp"
#include "detray/test/validation/svg_display.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// Actsvg include(s)
#include <actsvg/core.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s)
#include <array>
#include <string>

GTEST_TEST(svgtools, trajectories) {

    // This test creates the visualization using the illustrator class.
    // However, for full control over the process, it is also possible to use
    // the tools in svgstools::conversion, svgstools::display, and
    // actsvg::display by converting the object to a proto object, optionally
    // styling it, and then displaying it.

    // Creating the axes.
    const auto axes =
        actsvg::draw::x_y_axes("axes", {-250, 250}, {-250, 250},
                               actsvg::style::stroke(), "axis1", "axis2");

    // Creating the view.
    const actsvg::views::x_y view;

    // Creating the detector and geomentry context.
    vecmem::host_memory_resource host_mr;
    const auto [det, names] = detray::build_toy_detector(host_mr);
    using detector_t = decltype(det);

    detector_t::geometry_context gctx{};

    // Creating the illustrator.
    const detray::svgtools::illustrator il{det, names};

    // Show the relevant volumes in the detector.
    const auto [svg_volumes, _] =
        il.draw_volumes(std::vector{7u, 9u, 11u, 13u}, view);

    // Creating a ray.
    using algebra_t = typename detector_t::algebra_type;
    using vector3 = typename detector_t::vector3_type;

    const typename detector_t::point3_type ori{0.f, 0.f, 80.f};
    const typename detector_t::point3_type dir{0, 1.f, 1.f};

    const detray::detail::ray<algebra_t> ray(ori, 0.f, dir, 0.f);
    const auto ray_ir =
        detray::detector_scanner::run<detray::ray_scan>(gctx, det, ray);

    // Draw the trajectory.
    const auto svg_ray = il.draw_trajectory("trajectory", ray, 500.f, view);

    // Draw the intersections.
    auto ray_intersections = detray::detail::transcribe_intersections(ray_ir);
    const auto svg_ray_ir =
        il.draw_intersections("record", ray_intersections, ray.dir(), view);

    detray::svgtools::write_svg("test_svgtools_ray",
                                {svg_volumes, svg_ray, svg_ray_ir});

    // Creating a helix trajectory.

    // Constant magnetic field
    vector3 B{0.f * detray::unit<detray::scalar>::T,
              0.f * detray::unit<detray::scalar>::T,
              1.f * detray::unit<detray::scalar>::T};

    const detray::detail::helix<algebra_t> helix(
        ori, 0.f, detray::vector::normalize(dir), -8.f, &B);
    const auto helix_ir =
        detray::detector_scanner::run<detray::helix_scan>(gctx, det, helix);

    // Draw the trajectory.
    const auto svg_helix = il.draw_trajectory("trajectory", helix, 500.f, view);

    // Draw the intersections.
    auto helix_intersections =
        detray::detail::transcribe_intersections(helix_ir);
    const auto svg_helix_ir =
        il.draw_intersections("record", helix_intersections, helix.dir(), view);

    detray::svgtools::write_svg("test_svgtools_helix",
                                {svg_volumes, svg_helix, svg_helix_ir});
}
