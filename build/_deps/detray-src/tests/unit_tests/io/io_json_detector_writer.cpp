/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/units.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes/annulus2D.hpp"

// Detray IO include(s)
#include "detray/io/common/geometry_writer.hpp"
#include "detray/io/common/homogeneous_material_writer.hpp"
#include "detray/io/common/material_map_writer.hpp"
#include "detray/io/common/surface_grid_writer.hpp"
#include "detray/io/frontend/detector_writer.hpp"
#include "detray/io/json/json_writer.hpp"

// Detray test include(s)
#include "detray/test/utils/detectors/build_telescope_detector.hpp"
#include "detray/test/utils/detectors/build_toy_detector.hpp"
#include "detray/test/utils/detectors/build_wire_chamber.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s)
#include <gtest/gtest.h>

// System include(s)
#include <ios>

using namespace detray;

namespace {

// Use stereo annulus surfaces
constexpr scalar minR{7.2f * unit<scalar>::mm};
constexpr scalar maxR{12.0f * unit<scalar>::mm};
constexpr scalar minPhi{0.74195f};
constexpr scalar maxPhi{1.33970f};
constexpr scalar offset_x{-2.f * unit<scalar>::mm};
constexpr scalar offset_y{-2.f * unit<scalar>::mm};
mask<annulus2D> ann2{0u, minR, maxR, minPhi, maxPhi, offset_x, offset_y, 0.f};

// Surface positions
std::vector<scalar> positions = {0.f,   50.f,  100.f, 150.f, 200.f, 250.f,
                                 300.f, 350.f, 400.f, 450.f, 500.f};

tel_det_config<annulus2D> tel_cfg{ann2};

}  // anonymous namespace

/// Test the writing of a telescope detector geometry to json
GTEST_TEST(io, json_telescope_geometry_writer) {

    using detector_t = detector<telescope_metadata<annulus2D>>;

    // Telescope detector
    vecmem::host_memory_resource host_mr;
    auto [det, names] =
        build_telescope_detector(host_mr, tel_cfg.positions(positions));

    io::json_writer<detector_t, io::geometry_writer> geo_writer;
    geo_writer.write(det, names);
}

/// Test the writing of the toy detector material to json
GTEST_TEST(io, json_telescope_material_writer) {

    using detector_t = detector<telescope_metadata<annulus2D>>;

    // Telescope detector
    vecmem::host_memory_resource host_mr;

    auto [det, names] =
        build_telescope_detector(host_mr, tel_cfg.positions(positions));

    io::json_writer<detector_t, io::homogeneous_material_writer> mat_writer;
    mat_writer.write(det, names);
}

/// Test the writing of the toy detector grids to json
GTEST_TEST(io, json_toy_material_maps_writer) {

    using detector_t = detector<toy_metadata>;

    // Toy detector
    vecmem::host_memory_resource host_mr;
    toy_det_config toy_cfg{};
    toy_cfg.use_material_maps(true);
    auto [det, names] = build_toy_detector(host_mr, toy_cfg);

    io::json_writer<detector_t, io::material_map_writer> map_writer;
    map_writer.write(det, names,
                     std::ios::out | std::ios::binary | std::ios::trunc);
}

/// Test the writing of the toy detector grids to json
GTEST_TEST(io, json_toy_grid_writer) {

    using detector_t = detector<toy_metadata>;

    // Toy detector
    vecmem::host_memory_resource host_mr;
    auto [det, names] = build_toy_detector(host_mr);

    io::json_writer<detector_t, io::surface_grid_writer> grid_writer;
    grid_writer.write(det, names,
                      std::ios::out | std::ios::binary | std::ios::trunc);
}

/// Test the writing of the entire toy detector to json
GTEST_TEST(io, json_toy_detector_writer) {

    // Toy detector
    vecmem::host_memory_resource host_mr;
    toy_det_config toy_cfg{};
    toy_cfg.use_material_maps(true);
    const auto [det, names] = build_toy_detector(host_mr, toy_cfg);

    auto writer_cfg = io::detector_writer_config{}
                          .format(io::format::json)
                          .replace_files(true);
    io::write_detector(det, names, writer_cfg);
}

/// Test the writing of the entire wire chamber to json
GTEST_TEST(io, json_wire_chamber_writer) {

    // Wire chamber
    vecmem::host_memory_resource host_mr;
    wire_chamber_config<> wire_cfg{};
    auto [det, names] = build_wire_chamber(host_mr, wire_cfg);

    auto writer_cfg = io::detector_writer_config{}
                          .format(io::format::json)
                          .replace_files(true);
    io::write_detector(det, names, writer_cfg);
}
