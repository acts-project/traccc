/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/io/utils.hpp"
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
    const std::string cells_directory = "tml_full/ttbar_mu100/";

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Read the detector description.
    traccc::silicon_detector_description::host dd{host_mr};
    traccc::io::read_detector_description(
        dd, "tml_detector/trackml-detector.csv",
        "tml_detector/default-geometric-config-generic.json",
        traccc::data_format::csv);

    // Read csv file
    traccc::edm::silicon_cell_collection::host cells_csv(host_mr);
    traccc::io::read_cells(cells_csv, event, cells_directory,
                           traccc::getDummyLogger().clone(), &dd,
                           traccc::data_format::csv);

    // Write binary file
    traccc::io::write(event, cells_directory, traccc::data_format::binary,
                      vecmem::get_data(cells_csv));

    // Read binary file
    traccc::edm::silicon_cell_collection::host cells_binary(host_mr);
    traccc::io::read_cells(cells_binary, event, cells_directory,
                           traccc::getDummyLogger().clone(), &dd,
                           traccc::data_format::binary);

    // Delete binary file
    std::string io_cells_file =
        traccc::io::data_directory() + cells_directory +
        traccc::io::get_event_filename(event, "-cells.dat");
    std::remove(io_cells_file.c_str());

    EXPECT_TRUE(!std::ifstream(io_cells_file));

    // Check cells size
    EXPECT_GT(cells_csv.size(), 0u);
    EXPECT_EQ(cells_csv.size(), cells_binary.size());

    for (std::size_t i = 0; i < cells_csv.size(); i++) {
        EXPECT_EQ(cells_csv.channel0().at(i), cells_binary.channel0().at(i));
        EXPECT_EQ(cells_csv.channel1().at(i), cells_binary.channel1().at(i));
        EXPECT_EQ(cells_csv.activation().at(i),
                  cells_binary.activation().at(i));
        EXPECT_EQ(cells_csv.time().at(i), cells_binary.time().at(i));
        EXPECT_EQ(cells_csv.module_index().at(i),
                  cells_binary.module_index().at(i));
    }
}

// This defines the local frame test suite for binary spacepoint container
TEST(io_binary, spacepoint) {

    // Set event configuration
    const std::size_t event = 0;
    const std::string hits_directory = "tml_full/ttbar_mu200/";

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Read the detector description.
    traccc::silicon_detector_description::host dd{host_mr};
    traccc::io::read_detector_description(
        dd, "tml_detector/trackml-detector.csv",
        "tml_detector/default-geometric-config-generic.json",
        traccc::data_format::csv);

    // Read csv file
    traccc::measurement_collection_types::host measurements_csv(&host_mr);
    traccc::edm::spacepoint_collection::host spacepoints_csv(host_mr);
    traccc::io::read_spacepoints(spacepoints_csv, measurements_csv, event,
                                 hits_directory);

    // Write binary file
    traccc::io::write(event, hits_directory, traccc::data_format::binary,
                      vecmem::get_data(spacepoints_csv),
                      vecmem::get_data(measurements_csv));

    // Read binary file
    traccc::measurement_collection_types::host measurements_binary(&host_mr);
    traccc::edm::spacepoint_collection::host spacepoints_binary(host_mr);
    traccc::io::read_spacepoints(spacepoints_binary, measurements_binary, event,
                                 hits_directory, nullptr,
                                 traccc::data_format::binary);

    // Delete binary file
    std::string io_spacepoints_file =
        traccc::io::data_directory() + hits_directory +
        traccc::io::get_event_filename(event, "-hits.dat");
    std::remove(io_spacepoints_file.c_str());

    EXPECT_TRUE(!std::ifstream(io_spacepoints_file));

    // Check spacepoints size
    EXPECT_GT(spacepoints_csv.size(), 0);
    EXPECT_EQ(spacepoints_csv.size(), spacepoints_binary.size());

    for (std::size_t i = 0; i < spacepoints_csv.size(); i++) {
        EXPECT_EQ(spacepoints_csv[i], spacepoints_binary[i]);
    }
}

// This defines the local frame test suite for binary measurement container
TEST(io_binary, measurement) {
    // Set event configuration
    const std::size_t event = 0;
    const std::string measurements_directory = "tml_full/ttbar_mu300/";

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Read the detector description.
    traccc::silicon_detector_description::host dd{host_mr};
    traccc::io::read_detector_description(
        dd, "tml_detector/trackml-detector.csv",
        "tml_detector/default-geometric-config-generic.json",
        traccc::data_format::csv);

    // Read csv file
    traccc::measurement_collection_types::host measurements_csv(&host_mr);
    traccc::io::read_measurements(measurements_csv, event,
                                  measurements_directory);

    // Write binary file
    traccc::io::write(event, measurements_directory,
                      traccc::data_format::binary,
                      vecmem::get_data(measurements_csv));

    // Read binary file
    traccc::measurement_collection_types::host measurements_binary(&host_mr);
    traccc::io::read_measurements(measurements_binary, event,
                                  measurements_directory, nullptr,
                                  traccc::data_format::binary);

    // Delete binary file
    std::string io_measurements_file =
        traccc::io::data_directory() + measurements_directory +
        traccc::io::get_event_filename(event, "-measurements.dat");
    std::remove(io_measurements_file.c_str());

    EXPECT_TRUE(!std::ifstream(io_measurements_file));

    // Check header size
    EXPECT_GT(measurements_csv.size(), 0);
    EXPECT_EQ(measurements_csv.size(), measurements_binary.size());

    for (std::size_t i = 0; i < measurements_csv.size(); i++) {
        EXPECT_EQ(measurements_csv[i], measurements_binary[i]);
    }
}
