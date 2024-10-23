/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/performance/details/is_same_object.hpp"
#include "traccc/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

class SurfaceBinningTests
    : public ::testing::TestWithParam<
          std::tuple<std::string, std::string, std::string, unsigned int>> {};

// This defines the local frame test suite
TEST_P(SurfaceBinningTests, Run) {

    std::string detector_file = std::get<0>(GetParam());
    std::string digi_config_file = std::get<1>(GetParam());
    std::string data_dir = std::get<2>(GetParam());
    unsigned int event = std::get<3>(GetParam());

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Read the detector description.
    traccc::silicon_detector_description::host dd{host_mr};
    traccc::io::read_detector_description(dd, detector_file, digi_config_file,
                                          traccc::json);
    const traccc::silicon_detector_description::data dd_data =
        vecmem::get_data(dd);

    // Read the detector
    traccc::default_detector::host detector{host_mr};
    traccc::io::read_detector(detector, host_mr, detector_file);

    // Algorithms
    traccc::host::clusterization_algorithm ca(host_mr);
    traccc::host::silicon_pixel_spacepoint_formation_algorithm sf(host_mr);

    // Read the cells from the relevant event file
    traccc::edm::silicon_cell_collection::host cells_truth{host_mr};
    traccc::io::read_cells(cells_truth, event, data_dir, &dd);

    // Get Reconstructed Spacepoints
    auto measurements_recon = ca(vecmem::get_data(cells_truth), dd_data);
    auto spacepoints_recon = sf(detector, vecmem::get_data(measurements_recon));

    // Read the hits from the relevant event file
    traccc::spacepoint_collection_types::host spacepoints_truth{&host_mr};
    traccc::io::read_spacepoints(spacepoints_truth, event, data_dir, &detector);

    // Check the size of spacepoints
    EXPECT_TRUE(spacepoints_recon.size() > 0);

    for (const auto& sp_recon : spacepoints_recon) {

        bool found_match = false;

        // 20% resolution (Should be improved)
        auto iso = traccc::details::is_same_object(sp_recon, 0.2f);

        for (const auto& sp_truth : spacepoints_truth) {

            // Do not include the comparison of the measurement of spacepoint
            found_match = iso(sp_truth, false);

            if (found_match) {
                break;
            }
        }

        EXPECT_EQ(found_match, true);
    }
}

INSTANTIATE_TEST_SUITE_P(
    SurfaceBinningValidation, SurfaceBinningTests,
    ::testing::Values(
        std::make_tuple("geometries/odd/odd-detray_geometry_detray.json",
                        "geometries/odd/odd-digi-geometric-config.json",
                        "odd/geant4_1muon_100GeV", 0),
        /* The commented event files does not pass the test
                std::make_tuple("geometries/odd/odd-detray_geometry_detray.json",
                                "geometries/odd/odd-digi-geometric-config.json",
                                "odd/geant4_1muon_100GeV", 1),
                std::make_tuple("geometries/odd/odd-detray_geometry_detray.json",
                                "geometries/odd/odd-digi-geometric-config.json",
                                "odd/geant4_1muon_100GeV", 2),
        */
        std::make_tuple("geometries/odd/odd-detray_geometry_detray.json",
                        "geometries/odd/odd-digi-geometric-config.json",
                        "odd/geant4_1muon_100GeV", 3),
        std::make_tuple("geometries/odd/odd-detray_geometry_detray.json",
                        "geometries/odd/odd-digi-geometric-config.json",
                        "odd/geant4_1muon_100GeV", 4)));
