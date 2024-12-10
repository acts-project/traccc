/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/core/detector.hpp"

// Detray plugin include(s)
#include "detray/plugins/svgtools/illustrator.hpp"
#include "detray/plugins/svgtools/writer.hpp"

// Detray test include(s)
#include "detray/test/utils/detectors/build_toy_detector.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// Actsvg include(s)
#include <actsvg/core.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s)
#include <string>

GTEST_TEST(svgtools, landmarks) {

    // Axes.
    const auto axes =
        actsvg::draw::x_y_axes("axes", {-250, 250}, {-250, 250},
                               actsvg::style::stroke(), "axis1", "axis2");

    // Creating the view.
    const actsvg::views::x_y xy;
    const actsvg::views::z_r zr;

    // Creating the detector and geomentry context.
    vecmem::host_memory_resource host_mr;
    const auto [det, names] = detray::build_toy_detector(host_mr);
    using detector_t = decltype(det);

    using point = typename detector_t::point3_type;

    // Creating the illustrator class.
    const detray::svgtools::illustrator il{det, names};

    // Sometimes its useful to be able to just draw a point while debugging.
    // For this the draw_landmark function is available.
    const point test_point{100, 50, 20};

    const auto svg_xy = il.draw_landmark("landmark", test_point, xy);
    const auto svg_zr = il.draw_landmark("landmark", test_point, zr);
    detray::svgtools::write_svg("test_svgtools_landmark.svg",
                                {svg_xy, svg_zr, axes});
}
