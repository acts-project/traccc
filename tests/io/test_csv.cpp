/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/details/read_surfaces.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_particles.hpp"
#include "traccc/io/read_spacepoints_alt.hpp"

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
    auto& cells = single_module_cells.cells;
    auto& modules = single_module_cells.modules;
    ASSERT_EQ(cells.size(), 6u);
    ASSERT_EQ(modules.size(), 1u);
    auto module = single_module_cells.modules.at(0);

    ASSERT_EQ(module.module, 0u);
    ASSERT_EQ(cells.at(0).channel0, 123u);
    ASSERT_EQ(cells.at(0).channel1, 32u);
    ASSERT_EQ(cells.at(5).channel0, 174u);
    ASSERT_EQ(cells.at(5).channel1, 880u);
}

// This defines the local frame test suite
TEST_F(io, csv_read_two_modules) {

    auto two_module_cells = traccc::io::read_cells(
        get_datafile("two_modules/cells.csv"), traccc::data_format::csv);
    auto& cells = two_module_cells.cells;
    auto& modules = two_module_cells.modules;
    ASSERT_EQ(modules.size(), 2u);
    ASSERT_EQ(cells.size(), 14u);

    // Check cells in first module
    ASSERT_EQ(cells.at(0).channel0, 123u);
    ASSERT_EQ(cells.at(0).channel1, 32u);
    ASSERT_EQ(cells.at(0).module_link, 0u);
    ASSERT_EQ(cells.at(5).channel0, 174u);
    ASSERT_EQ(cells.at(5).channel1, 880u);
    ASSERT_EQ(cells.at(5).module_link, 0u);

    ASSERT_EQ(modules.at(0u).module, 0u);

    // Check cells in second module
    ASSERT_EQ(cells.at(6).channel0, 0u);
    ASSERT_EQ(cells.at(6).channel1, 4u);
    ASSERT_EQ(cells.at(6).module_link, 1u);
    ASSERT_EQ(cells.at(13).channel0, 5u);
    ASSERT_EQ(cells.at(13).channel1, 98u);
    ASSERT_EQ(cells.at(13).module_link, 1u);

    ASSERT_EQ(modules.at(1u).module, 1u);
}

// This reads in the tml pixel barrel first event
TEST_F(io, csv_read_tml_transforms) {
    std::string file = get_datafile("tml_detector/trackml-detector.csv");

    auto tml_barrel_transforms = traccc::io::details::read_surfaces(file);

    ASSERT_EQ(tml_barrel_transforms.size(), 18791u);
}

// This reads in the tml pixel barrel first event
TEST_F(io, csv_read_tml_pixelbarrel) {

    auto tml_barrel_modules =
        traccc::io::read_cells(
            get_datafile("tml_pixel_barrel/event000000000-cells.csv"),
            traccc::data_format::csv)
            .modules;

    ASSERT_EQ(tml_barrel_modules.size(), 2382u);
}

// This checks if hit and measurement container from the first single muon event
TEST_F(io, csv_read_tml_single_muon) {
    vecmem::host_memory_resource resource;

    // Read the surface transforms
    auto surface_transforms =
        traccc::io::read_geometry("tml_detector/trackml-detector.csv");

    // Read the hits from the relevant event file
    auto spacepoints_per_event = traccc::io::read_spacepoints_alt(
        0, "tml_full/single_muon/", surface_transforms,
        traccc::data_format::csv, &resource);

    // Read the measurements from the relevant event file
    auto measurements_per_event = traccc::io::read_measurements(
        0, "tml_full/single_muon/", traccc::data_format::csv, &resource);

    // Read the particles from the relevant event file
    traccc::particle_collection_types::host particles_per_event =
        traccc::io::read_particles(0, "tml_full/single_muon/",
                                   traccc::data_format::csv, &resource);

    ASSERT_EQ(spacepoints_per_event.modules.size(), 11u);
    ASSERT_EQ(measurements_per_event.modules.size(), 11u);

    ASSERT_EQ(spacepoints_per_event.spacepoints.size(), 11u);
    ASSERT_EQ(measurements_per_event.measurements.size(), 11u);

    ASSERT_EQ(particles_per_event.size(), 1u);
}