/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation_algorithm.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/read_spacepoints.hpp"

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
                                          traccc::csv);
    const traccc::silicon_detector_description::data dd_data =
        vecmem::get_data(dd);

    // Algorithms
    traccc::host::clusterization_algorithm ca(host_mr);
    traccc::host::spacepoint_formation_algorithm sf(host_mr);

    // Read the cells from the relevant event file
    traccc::cell_collection_types::host cells_truth{&host_mr};
    traccc::io::read_cells(cells_truth, event, data_dir, &dd);

    // Get Reconstructed Spacepoints
    auto measurements_recon = ca(vecmem::get_data(cells_truth), dd_data);
    auto spacepoints_recon = sf(vecmem::get_data(measurements_recon), dd_data);

    // Read the hits from the relevant event file
    traccc::spacepoint_collection_types::host spacepoints_truth{&host_mr};
    traccc::io::read_spacepoints(spacepoints_truth, event, data_dir, &dd);

    // Check the size of spacepoints
    EXPECT_TRUE(spacepoints_recon.size() > 0);
    EXPECT_EQ(spacepoints_recon.size(), spacepoints_truth.size());

    for (std::size_t i = 0; i < spacepoints_recon.size(); i++) {

        const auto& sp_recon = spacepoints_recon[i];
        const auto& sp_truth = spacepoints_truth[i];

        // Check that the spacepoints belong to the same module
        EXPECT_EQ(sp_recon.meas.module_link, sp_truth.meas.module_link);

        // Make sure that the difference in spacepoint position is less than
        // 1%
        EXPECT_TRUE(traccc::getter::norm(sp_recon.global - sp_truth.global) /
                        traccc::getter::norm(sp_recon.global) <
                    0.01);
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
