/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/reader.hpp"
#include "traccc/io/write.hpp"

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

    // Read the digitization configuration file
    auto digi_cfg = traccc::read_digitization_config(
        "tml_detector/default-geometric-config-generic.json");

    // Read csv file
    traccc::cell_container_types::host cells_csv =
        traccc::read_cells_from_event(event, cells_directory,
                                      traccc::data_format::csv,
                                      surface_transforms, digi_cfg, host_mr);

    // Write binary file
    traccc::io::write(event, cells_directory, traccc::data_format::binary,
                      traccc::get_data(cells_csv));

    // Read binary file
    traccc::cell_container_types::host cells_binary =
        traccc::read_cells_from_event(event, cells_directory,
                                      traccc::data_format::binary,
                                      surface_transforms, digi_cfg, host_mr);

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
    traccc::spacepoint_container_types::host spacepoints_csv =
        traccc::read_spacepoints_from_event(event, hits_directory,
                                            traccc::data_format::csv,
                                            surface_transforms, host_mr);

    // Write binary file
    traccc::io::write(event, hits_directory, traccc::data_format::binary,
                      traccc::get_data(spacepoints_csv));

    // Read binary file
    traccc::spacepoint_container_types::host spacepoints_binary =
        traccc::read_spacepoints_from_event(event, hits_directory,
                                            traccc::data_format::binary,
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

// This defines the local frame test suite for binary measurement container
TEST(io_binary, measurement) {

    // Set event configuration
    const std::size_t event = 0;
    const std::string measurements_directory = "tml_full/ttbar_mu200/";

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Read csv file
    traccc::measurement_container_types::host measurements_csv =
        traccc::read_measurements_from_event(event, measurements_directory,
                                             traccc::data_format::csv, host_mr);

    // Write binary file
    traccc::io::write(event, measurements_directory,
                      traccc::data_format::binary,
                      traccc::get_data(measurements_csv));

    // Read binary file
    traccc::measurement_container_types::host measurements_binary =
        traccc::read_measurements_from_event(event, measurements_directory,
                                             traccc::data_format::binary,
                                             host_mr);

    // Delete binary file
    std::string io_measurements_file =
        traccc::data_directory() + measurements_directory +
        traccc::get_event_filename(event, "-measurements.dat");
    std::remove(io_measurements_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_measurements_file));

    // Check header size
    ASSERT_TRUE(measurements_csv.size() > 0);
    ASSERT_EQ(measurements_csv.size(), measurements_binary.size());

    auto& headers_csv = measurements_csv.get_headers();
    auto& headers_bin = measurements_binary.get_headers();

    for (std::size_t i = 0; i < measurements_csv.size(); i++) {

        // Check header content
        ASSERT_EQ(headers_csv[i], headers_bin[i]);

        auto& items_csv = measurements_csv.get_items()[i];
        auto& items_bin = measurements_binary.get_items()[i];

        // Check item size
        ASSERT_TRUE(items_csv.size() > 0);
        ASSERT_EQ(items_csv.size(), items_bin.size());

        // Check item contents
        for (std::size_t j = 0; j < items_csv.size(); j++) {
            ASSERT_EQ(items_csv[j], items_bin[j]);
        }
    }
}