/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/experimental/spacepoint_formation.hpp"

// Detray include(s).
#include "detray/detectors/create_telescope_detector.hpp"
#include "detray/intersection/detail/trajectories.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

using namespace traccc;

TEST(spacepoint_formation, cpu) {

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Use rectangle surfaces
    detray::mask<detray::rectangle2D<>> rectangle{
        0u, 10000.f * detray::unit<scalar>::mm,
        10000.f * detray::unit<scalar>::mm};

    // Plane alignment direction (aligned to x-axis)
    detray::detail::ray<transform3> traj{{0, 0, 0}, 0, {1, 0, 0}, -1};
    // Position of planes (in mm unit)
    std::vector<scalar> plane_positions = {20.f,  40.f,  60.f,  80.f, 100.f,
                                           120.f, 140.f, 160.f, 180.f};

    detray::tel_det_config<> tel_cfg{rectangle};
    tel_cfg.positions(plane_positions);
    tel_cfg.pilot_track(traj);

    // Create telescope geometry
    const auto [det, name_map] = create_telescope_detector(host_mr, tel_cfg);

    // Surface lookup
    auto surfaces = det.surface_lookup();

    // Prepare measurement collection
    typename measurement_collection_types::host measurements{&host_mr};

    // Add a measurement at the first plane
    measurements.push_back({{7.f, 2.f}, {0.f, 0.f}, surfaces[0].barcode()});

    // Add a measurement at the last plane
    measurements.push_back({{10.f, 15.f}, {0.f, 0.f}, surfaces[8u].barcode()});

    // Run spacepoint formation
    experimental::spacepoint_formation<decltype(det)> sp_formation(host_mr);
    auto spacepoints = sp_formation(det, measurements);

    // Check the results
    EXPECT_EQ(spacepoints.size(), 2u);
    EXPECT_FLOAT_EQ(spacepoints[0].global[0], 20.f);
    EXPECT_FLOAT_EQ(spacepoints[0].global[1], 7.f);
    EXPECT_FLOAT_EQ(spacepoints[0].global[2], 2.f);
    EXPECT_FLOAT_EQ(spacepoints[1].global[0], 180.f);
    EXPECT_FLOAT_EQ(spacepoints[1].global[1], 10.f);
    EXPECT_FLOAT_EQ(spacepoints[1].global[2], 15.f);
}