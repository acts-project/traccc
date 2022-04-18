/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/reader.hpp"
#include "traccc/io/writer.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System
#include <cstdio>
#include <fstream>

// This defines the local frame test suite for binary cell container
TEST(io_binary, cell) {

    // Set event configuration
    const std::size_t event = 0;
    const std::string cells_directory = "tml_full/ttbar_mu200/";

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Read the surface transforms
    auto surface_transforms =
        traccc::read_geometry("tml_detector/trackml-detector.csv");

    // Read csv file
    traccc::host_cell_container cells_csv = traccc::read_cells_from_event(
        event, cells_directory, "csv", surface_transforms, host_mr);

    // Write binary file
    traccc::write_cells(event, cells_directory, "binary", cells_csv);

    // Read binary file
    traccc::host_cell_container cells_binary = traccc::read_cells_from_event(
        event, cells_directory, "binary", surface_transforms, host_mr);

    // Delete binary file
    std::string io_cells_file = traccc::data_directory() + cells_directory +
                                traccc::get_event_filename(event, "-cells.dat");
    std::remove(io_cells_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_cells_file));

    // Check header size
    ASSERT_TRUE(cells_csv.size() > 0);
    ASSERT_EQ(cells_csv.size(), cells_binary.size());

    auto& headers_csv = cells_csv.get_headers();
    auto& headers_bin = cells_binary.get_headers();

    for (std::size_t i = 0; i < cells_csv.size(); i++) {

        // Check header content
        ASSERT_EQ(headers_csv[i].module, headers_bin[i].module);
        ASSERT_EQ(headers_csv[i].placement, headers_bin[i].placement);

        auto& items_csv = cells_csv.get_items()[i];
        auto& items_bin = cells_binary.get_items()[i];

        // Check item size
        ASSERT_TRUE(items_csv.size() > 0);
        ASSERT_EQ(items_csv.size(), items_bin.size());

        // Check item contents
        for (std::size_t j = 0; j < items_csv.size(); j++) {
            ASSERT_EQ(items_csv[j], items_bin[j]);
        }
    }
}

// This defines the local frame test suite for binary spacepoint container
TEST(io_binary, spacepoint) {

    // Set event configuration
    const std::size_t event = 0;
    const std::string hits_directory = "tml_full/ttbar_mu200/";

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Read the surface transforms
    auto surface_transforms =
        traccc::read_geometry("tml_detector/trackml-detector.csv");

    // Read csv file
    traccc::host_spacepoint_container spacepoints_csv =
        traccc::read_spacepoints_from_event(event, hits_directory, "csv",
                                            surface_transforms, host_mr);

    // Write binary file
    traccc::write_spacepoints(event, hits_directory, "binary", spacepoints_csv);

    // Read binary file
    traccc::host_spacepoint_container spacepoints_binary =
        traccc::read_spacepoints_from_event(event, hits_directory, "binary",
                                            surface_transforms, host_mr);

    // Delete binary file
    std::string io_spacepoints_file =
        traccc::data_directory() + hits_directory +
        traccc::get_event_filename(event, "-hits.dat");
    std::remove(io_spacepoints_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_spacepoints_file));

    // Check header size
    ASSERT_TRUE(spacepoints_csv.size() > 0);
    ASSERT_EQ(spacepoints_csv.size(), spacepoints_binary.size());

    auto& headers_csv = spacepoints_csv.get_headers();
    auto& headers_bin = spacepoints_binary.get_headers();

    for (std::size_t i = 0; i < spacepoints_csv.size(); i++) {

        // Check header content
        ASSERT_EQ(headers_csv[i], headers_bin[i]);

        auto& items_csv = spacepoints_csv.get_items()[i];
        auto& items_bin = spacepoints_binary.get_items()[i];

        // Check item size
        ASSERT_TRUE(items_csv.size() > 0);
        ASSERT_EQ(items_csv.size(), items_bin.size());

        // Check item contents
        for (std::size_t j = 0; j < items_csv.size(); j++) {
            ASSERT_EQ(items_csv[j], items_bin[j]);
        }
    }
}