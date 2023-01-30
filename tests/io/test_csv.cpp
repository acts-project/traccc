/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/details/read_surfaces.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_cells_alt.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_particles.hpp"
#include "traccc/io/read_spacepoints.hpp"

// Test include(s).
#include "tests/data_test.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

class io : public traccc::tests::data_test {};

// This defines the local frame test suite
TEST_F(io, csv_read_single_module) {

    auto single_module_cells = traccc::io::read_cells(
        get_datafile("single_module/cells.csv"), traccc::data_format::csv);
    ASSERT_EQ(single_module_cells.size(), 1u);
    auto module_cells = single_module_cells.at(0).items;
    auto module = single_module_cells.at(0).header;

    ASSERT_EQ(module.module, 0u);
    ASSERT_EQ(module_cells.size(), 6u);
    ASSERT_EQ(module_cells.at(0).channel0, 123u);
    ASSERT_EQ(module_cells.at(0).channel1, 32u);
    ASSERT_EQ(module_cells.at(5).channel0, 174u);
    ASSERT_EQ(module_cells.at(5).channel1, 880u);
}

// This defines the local frame test suite
TEST_F(io, csv_read_two_modules) {

    auto two_module_cells = traccc::io::read_cells(
        get_datafile("two_modules/cells.csv"), traccc::data_format::csv);
    ASSERT_EQ(two_module_cells.size(), 2u);

    auto first_module_cells = two_module_cells.at(0).items;
    auto first_module = two_module_cells.at(0).header;

    ASSERT_EQ(first_module_cells.size(), 6u);
    ASSERT_EQ(first_module_cells.at(0).channel0, 123u);
    ASSERT_EQ(first_module_cells.at(0).channel1, 32u);
    ASSERT_EQ(first_module_cells.at(5).channel0, 174u);
    ASSERT_EQ(first_module_cells.at(5).channel1, 880u);

    ASSERT_EQ(first_module.module, 0u);

    auto second_module_cells = two_module_cells.at(1).items;
    auto second_module = two_module_cells.at(1).header;

    ASSERT_EQ(second_module_cells.size(), 8u);
    ASSERT_EQ(second_module_cells.at(0).channel0, 0u);
    ASSERT_EQ(second_module_cells.at(0).channel1, 4u);
    ASSERT_EQ(second_module_cells.at(7).channel0, 5u);
    ASSERT_EQ(second_module_cells.at(7).channel1, 98u);

    ASSERT_EQ(second_module.module, 1u);
}

// This defines the local frame test suite
TEST_F(io, csv_read_cells_alt) {

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

    // Read csv file to cell item + module header container
    traccc::cell_container_types::host cells_csv =
        traccc::io::read_cells(event, cells_directory, traccc::data_format::csv,
                               &surface_transforms, &digi_cfg, &host_mr);

    // Read csv file to collection of cells + collection of modules
    traccc::alt_cell_reader_output_t alt_cells_modules_csv =
        traccc::io::read_cells_alt(event, cells_directory,
                                   traccc::data_format::csv,
                                   &surface_transforms, &digi_cfg, &host_mr);

    const traccc::alt_cell_collection_types::host& alt_cells_csv =
        alt_cells_modules_csv.cells;
    const traccc::cell_module_collection_types::host& alt_modules_csv =
        alt_cells_modules_csv.modules;

    std::size_t k = 0;
    for (std::size_t i = 0; i < cells_csv.size(); i++) {
        const traccc::cell_module& cm = cells_csv.get_headers()[i];
        for (std::size_t j = 0; j < cells_csv.get_items()[i].size(); ++j) {
            const traccc::cell& c = cells_csv.get_items()[i][j];
            if (k < alt_cells_csv.size()) {
                EXPECT_EQ(cm, alt_modules_csv.at(alt_cells_csv[k].module_link));
                EXPECT_EQ(c, alt_cells_csv[k].c);
            }
            k++;
        }
    }
    EXPECT_EQ(k, alt_cells_csv.size());
}

// This reads in the tml pixel barrel first event
TEST_F(io, csv_read_tml_transforms) {
    std::string file = get_datafile("tml_detector/trackml-detector.csv");

    auto tml_barrel_transforms = traccc::io::details::read_surfaces(file);

    ASSERT_EQ(tml_barrel_transforms.size(), 18791u);
}

// This reads in the tml pixel barrel first event
TEST_F(io, csv_read_tml_pixelbarrel) {

    auto tml_barrel_modules = traccc::io::read_cells(
        get_datafile("tml_pixel_barrel/event000000000-cells.csv"),
        traccc::data_format::csv);

    ASSERT_EQ(tml_barrel_modules.size(), 2382u);
}

// This checks if hit and measurement container from the first single muon event
TEST_F(io, csv_read_tml_single_muon) {
    vecmem::host_memory_resource resource;

    // Read the surface transforms
    auto surface_transforms =
        traccc::io::read_geometry("tml_detector/trackml-detector.csv");

    // Read the hits from the relevant event file
    traccc::spacepoint_container_types::host spacepoints_per_event =
        traccc::io::read_spacepoints(0, "tml_full/single_muon/",
                                     surface_transforms,
                                     traccc::data_format::csv, &resource);

    // Read the measurements from the relevant event file
    traccc::measurement_container_types::host measurements_per_event =
        traccc::io::read_measurements(0, "tml_full/single_muon/",
                                      traccc::data_format::csv, &resource);

    // Read the particles from the relevant event file
    traccc::particle_collection_types::host particles_per_event =
        traccc::io::read_particles(0, "tml_full/single_muon/",
                                   traccc::data_format::csv, &resource);

    const auto sp_header_size = spacepoints_per_event.size();
    const auto ms_header_size = measurements_per_event.size();
    ASSERT_EQ(sp_header_size, 11u);
    ASSERT_EQ(ms_header_size, 11u);

    for (std::size_t i = 0; i < sp_header_size; i++) {
        ASSERT_EQ(spacepoints_per_event[i].items.size(), 1u);
    }
    for (std::size_t i = 0; i < ms_header_size; i++) {
        ASSERT_EQ(measurements_per_event[i].items.size(), 1u);
    }

    ASSERT_EQ(particles_per_event.size(), 1u);
}