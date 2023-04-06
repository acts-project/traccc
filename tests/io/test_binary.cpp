/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints_alt.hpp"
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
    const std::string cells_directory = "tml_full/ttbar_mu200/";

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Read the surface transforms
    auto surface_transforms =
        traccc::io::read_geometry("tml_detector/trackml-detector.csv");

    // Read the digitization configuration file
    auto digi_cfg = traccc::io::read_digitization_config(
        "tml_detector/default-geometric-config-generic.json");

    // Read csv file
    auto reader_csv =
        traccc::io::read_cells(event, cells_directory, traccc::data_format::csv,
                               &surface_transforms, &digi_cfg, &host_mr);
    const traccc::cell_collection_types::host& cells_csv = reader_csv.cells;
    const traccc::cell_module_collection_types::host& modules_csv =
        reader_csv.modules;

    // Write binary file
    traccc::io::write(event, cells_directory, traccc::data_format::binary,
                      vecmem::get_data(cells_csv),
                      vecmem::get_data(modules_csv));

    // Read binary file
    auto reader_binary = traccc::io::read_cells(
        event, cells_directory, traccc::data_format::binary,
        &surface_transforms, &digi_cfg, &host_mr);
    const traccc::cell_collection_types::host& cells_binary =
        reader_binary.cells;
    const traccc::cell_module_collection_types::host& modules_binary =
        reader_binary.modules;

    // Delete binary file
    std::string io_cells_file =
        traccc::io::data_directory() + cells_directory +
        traccc::io::get_event_filename(event, "-cells.dat");
    std::remove(io_cells_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_cells_file));

    std::string io_modules_file =
        traccc::io::data_directory() + cells_directory +
        traccc::io::get_event_filename(event, "-modules.dat");
    std::remove(io_modules_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_modules_file));

    // Check cells size
    ASSERT_TRUE(cells_csv.size() > 0);
    ASSERT_EQ(cells_csv.size(), cells_binary.size());

    // Check modules size
    ASSERT_TRUE(modules_csv.size() > 0);
    ASSERT_EQ(modules_csv.size(), modules_binary.size());

    for (std::size_t i = 0; i < cells_csv.size(); i++) {
        ASSERT_EQ(cells_csv[i], cells_binary[i]);
    }
    for (std::size_t i = 0; i < modules_csv.size(); i++) {
        ASSERT_EQ(modules_csv[i].module, modules_binary[i].module);
        ASSERT_EQ(modules_csv[i].placement, modules_binary[i].placement);
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
        traccc::io::read_geometry("tml_detector/trackml-detector.csv");

    // Read csv file
    auto reader_csv = traccc::io::read_spacepoints_alt(
        event, hits_directory, surface_transforms, traccc::data_format::csv,
        &host_mr);
    const traccc::spacepoint_collection_types::host& spacepoints_csv =
        reader_csv.spacepoints;
    const traccc::cell_module_collection_types::host& modules_csv =
        reader_csv.modules;

    // // Write binary file
    traccc::io::write(event, hits_directory, traccc::data_format::binary,
                      vecmem::get_data(spacepoints_csv),
                      vecmem::get_data(modules_csv));

    // Read binary file
    auto reader_binary = traccc::io::read_spacepoints_alt(
        event, hits_directory, surface_transforms, traccc::data_format::binary,
        &host_mr);
    const traccc::spacepoint_collection_types::host& spacepoints_binary =
        reader_binary.spacepoints;
    const traccc::cell_module_collection_types::host& modules_binary =
        reader_binary.modules;

    // Delete binary file
    std::string io_spacepoints_file =
        traccc::io::data_directory() + hits_directory +
        traccc::io::get_event_filename(event, "-hits.dat");
    std::remove(io_spacepoints_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_spacepoints_file));

    std::string io_modules_file =
        traccc::io::data_directory() + hits_directory +
        traccc::io::get_event_filename(event, "-modules.dat");
    std::remove(io_modules_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_modules_file));

    // Check spacepoints size
    ASSERT_TRUE(spacepoints_csv.size() > 0);
    ASSERT_EQ(spacepoints_csv.size(), spacepoints_binary.size());

    // Check modules size
    ASSERT_TRUE(modules_csv.size() > 0);
    ASSERT_EQ(modules_csv.size(), modules_binary.size());

    for (std::size_t i = 0; i < spacepoints_csv.size(); i++) {
        ASSERT_EQ(spacepoints_csv[i], spacepoints_binary[i]);
    }
    for (std::size_t i = 0; i < modules_csv.size(); i++) {
        ASSERT_EQ(modules_csv[i], modules_binary[i]);
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
    auto reader_csv = traccc::io::read_measurements(
        event, measurements_directory, traccc::data_format::csv, &host_mr);
    const traccc::alt_measurement_collection_types::host& measurements_csv =
        reader_csv.measurements;
    const traccc::cell_module_collection_types::host& modules_csv =
        reader_csv.modules;

    // Write binary file
    traccc::io::write(
        event, measurements_directory, traccc::data_format::binary,
        vecmem::get_data(measurements_csv), vecmem::get_data(modules_csv));

    // Read binary file
    auto reader_binary = traccc::io::read_measurements(
        event, measurements_directory, traccc::data_format::binary, &host_mr);
    const traccc::alt_measurement_collection_types::host& measurements_binary =
        reader_binary.measurements;
    const traccc::cell_module_collection_types::host& modules_binary =
        reader_binary.modules;

    // Delete binary file
    std::string io_measurements_file =
        traccc::io::data_directory() + measurements_directory +
        traccc::io::get_event_filename(event, "-measurements.dat");
    std::remove(io_measurements_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_measurements_file));

    std::string io_modules_file =
        traccc::io::data_directory() + measurements_directory +
        traccc::io::get_event_filename(event, "-modules.dat");
    std::remove(io_modules_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_modules_file));

    // Check header size
    ASSERT_TRUE(measurements_csv.size() > 0);
    ASSERT_EQ(measurements_csv.size(), measurements_binary.size());

    // Check modules size
    ASSERT_TRUE(modules_csv.size() > 0);
    ASSERT_EQ(modules_csv.size(), modules_binary.size());

    for (std::size_t i = 0; i < measurements_csv.size(); i++) {
        ASSERT_EQ(measurements_csv[i], measurements_binary[i]);
    }
    for (std::size_t i = 0; i < modules_csv.size(); i++) {
        ASSERT_EQ(modules_csv[i].module, modules_binary[i].module);
    }
}