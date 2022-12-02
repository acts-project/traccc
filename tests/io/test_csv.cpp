/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/details/read_surfaces.hpp"
#include "traccc/io/read_cells.hpp"
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