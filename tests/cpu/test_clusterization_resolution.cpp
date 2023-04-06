/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_spacepoints_alt.hpp"

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
    auto surface_transforms = traccc::io::read_geometry(detector_file);

    // Read the digitization configuration file
    auto digi_cfg = traccc::io::read_digitization_config(digi_config_file);

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Algorithms
    traccc::clusterization_algorithm ca(host_mr);
    traccc::spacepoint_formation sf(host_mr);

    // Read the cells from the relevant event file
    auto readOut =
        traccc::io::read_cells(event, data_dir, traccc::data_format::csv,
                               &surface_transforms, &digi_cfg, &host_mr);

    const traccc::cell_collection_types::host& cells_truth = readOut.cells;
    const traccc::cell_module_collection_types::host& modules = readOut.modules;

    // Get Reconstructed Spacepoints
    auto measurements_recon = ca(cells_truth, modules);
    auto spacepoints_recon = sf(measurements_recon, modules);

    // Read the hits from the relevant event file
    auto sp_readOut =
        traccc::io::read_spacepoints_alt(event, data_dir, surface_transforms,
                                         traccc::data_format::csv, &host_mr);

    const traccc::spacepoint_collection_types::host& spacepoints_truth =
        sp_readOut.spacepoints;
    const traccc::cell_module_collection_types::host& modules_2 =
        sp_readOut.modules;

    // Check the size of spacepoints
    EXPECT_TRUE(spacepoints_recon.size() > 0);
    EXPECT_EQ(spacepoints_recon.size(), spacepoints_truth.size());

    for (std::size_t i = 0; i < spacepoints_recon.size(); i++) {

        const auto& sp_recon = spacepoints_recon[i];
        const auto& sp_truth = spacepoints_truth[i];

        // Check that the spacepoints belong to the same module
        EXPECT_EQ(modules.at(sp_recon.meas.module_link).module,
                  modules_2.at(sp_truth.meas.module_link).module);

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
