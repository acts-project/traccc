/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/io/reader.hpp"

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

    // Read the surface transforms
    auto surface_transforms = traccc::read_geometry(detector_file);

    // Read the digitization configuration file
    auto digi_cfg = traccc::read_digitization_config(digi_config_file);

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Algorithms
    traccc::clusterization_algorithm ca(host_mr);
    traccc::spacepoint_formation sf(host_mr);

    // Read the cells from the relevant event file
    traccc::cell_container_types::host cells_truth =
        traccc::read_cells_from_event(event, data_dir, traccc::data_format::csv,
                                      surface_transforms, digi_cfg, host_mr);

    // Get Reconstructed Spacepoints
    auto measurements_recon = ca(cells_truth);
    auto spacepoints_recon = sf(measurements_recon);

    // Read the hits from the relevant event file
    traccc::spacepoint_container_types::host spacepoints_truth =
        traccc::read_spacepoints_from_event(event, data_dir,
                                            traccc::data_format::csv,
                                            surface_transforms, host_mr);

    // Check the size of spacepoints
    EXPECT_TRUE(spacepoints_recon.size() > 0);
    EXPECT_EQ(spacepoints_recon.size(), spacepoints_truth.size());
    EXPECT_TRUE(spacepoints_recon.total_size() > 0);
    EXPECT_EQ(spacepoints_recon.total_size(), spacepoints_truth.total_size());

    for (std::size_t i = 0; i < spacepoints_recon.size(); i++) {
        EXPECT_EQ(spacepoints_recon.at(i).header,
                  spacepoints_truth.at(i).header);

        // Get the spacepoint vectors
        traccc::spacepoint_collection_types::host& sp_collection_recon =
            spacepoints_recon[i].items;

        traccc::spacepoint_collection_types::host& sp_collection_truth =
            spacepoints_truth[i].items;

        EXPECT_EQ(sp_collection_recon.size(), sp_collection_truth.size());

        // Iterate over each spacepoint
        for (std::size_t j = 0; j < sp_collection_recon.size(); j++) {
            const auto& sp_recon = sp_collection_recon[j];
            const auto& sp_truth = sp_collection_truth[j];

            // Make sure that the difference in spacepoint position is less than
            // 1%
            EXPECT_TRUE(
                traccc::getter::norm(sp_recon.global - sp_truth.global) /
                    traccc::getter::norm(sp_recon.global) <
                0.01);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SurfaceBinningValidation, SurfaceBinningTests,
    ::testing::Values(
        std::make_tuple("tml_detector/trackml-detector.csv",
                        "tml_detector/default-geometric-config-generic.json",
                        "tml_full/single_muon/", 0),
        std::make_tuple("tml_detector/trackml-detector.csv",
                        "tml_detector/default-geometric-config-generic.json",
                        "tml_full/single_muon/", 1),
        std::make_tuple("tml_detector/trackml-detector.csv",
                        "tml_detector/default-geometric-config-generic.json",
                        "tml_full/single_muon/", 2),
        std::make_tuple("tml_detector/trackml-detector.csv",
                        "tml_detector/default-geometric-config-generic.json",
                        "tml_full/single_muon/", 3),
        std::make_tuple("tml_detector/trackml-detector.csv",
                        "tml_detector/default-geometric-config-generic.json",
                        "tml_full/single_muon/", 4),
        std::make_tuple("tml_detector/trackml-detector.csv",
                        "tml_detector/default-geometric-config-generic.json",
                        "tml_full/single_muon/", 5),
        std::make_tuple("tml_detector/trackml-detector.csv",
                        "tml_detector/default-geometric-config-generic.json",
                        "tml_full/single_muon/", 6),
        std::make_tuple("tml_detector/trackml-detector.csv",
                        "tml_detector/default-geometric-config-generic.json",
                        "tml_full/single_muon/", 7),
        std::make_tuple("tml_detector/trackml-detector.csv",
                        "tml_detector/default-geometric-config-generic.json",
                        "tml_full/single_muon/", 8),
        std::make_tuple("tml_detector/trackml-detector.csv",
                        "tml_detector/default-geometric-config-generic.json",
                        "tml_full/single_muon/", 9)));
