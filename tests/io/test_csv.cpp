/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/details/read_surfaces.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/read_digitization_config.hpp"
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

    vecmem::host_memory_resource resource;

    traccc::cell_collection_types::host cells{&resource};
    traccc::io::read_cells(cells, get_datafile("single_module/cells.csv"));

    ASSERT_EQ(cells.size(), 6u);

    EXPECT_EQ(cells.at(0).channel0, 123u);
    EXPECT_EQ(cells.at(0).channel1, 32u);
    EXPECT_FLOAT_EQ(static_cast<float>(cells.at(0).activation), 0.002f);
    EXPECT_EQ(cells.at(0).module_link, 0u);

    EXPECT_EQ(cells.at(5).channel0, 174u);
    EXPECT_EQ(cells.at(5).channel1, 880u);
    EXPECT_FLOAT_EQ(static_cast<float>(cells.at(5).activation), 0.23f);
    EXPECT_EQ(cells.at(5).module_link, 0u);
}

// This defines the local frame test suite
TEST_F(io, csv_read_two_modules) {

    vecmem::host_memory_resource resource;

    traccc::cell_collection_types::host cells{&resource};
    traccc::io::read_cells(cells, get_datafile("two_modules/cells.csv"));

    ASSERT_EQ(cells.size(), 14u);

    // Check cells in first module
    EXPECT_EQ(cells.at(0).channel0, 123u);
    EXPECT_EQ(cells.at(0).channel1, 32u);
    EXPECT_EQ(cells.at(5).channel0, 174u);
    EXPECT_EQ(cells.at(5).channel1, 880u);

    // Check cells in second module
    EXPECT_EQ(cells.at(6).channel0, 0u);
    EXPECT_EQ(cells.at(6).channel1, 4u);
    EXPECT_EQ(cells.at(13).channel0, 5u);
    EXPECT_EQ(cells.at(13).channel1, 98u);
}

// This reads in the tml pixel barrel first event
TEST_F(io, csv_read_tml_transforms) {
    std::string file = get_datafile("tml_detector/trackml-detector.csv");

    auto tml_barrel_transforms = traccc::io::details::read_surfaces(file);

    ASSERT_EQ(tml_barrel_transforms.size(), 18791u);
}

// This reads in the tml pixel barrel first event
TEST_F(io, csv_read_tml_pixelbarrel) {

    vecmem::host_memory_resource resource;

    traccc::cell_collection_types::host cells{&resource};
    traccc::io::read_cells(
        cells, get_datafile("tml_pixel_barrel/event000000000-cells.csv"),
        nullptr, traccc::data_format::csv, false);

    EXPECT_EQ(cells.size(), 179961u);
}

// This checks if hit and measurement container from the first single muon event
TEST_F(io, csv_read_tml_single_muon) {
    vecmem::host_memory_resource resource;

    // Read the detector description.
    traccc::detector_description::host dd{resource};
    traccc::io::read_detector_description(
        dd, "tml_detector/trackml-detector.csv",
        "tml_detector/default-geometric-config-generic.json",
        traccc::data_format::csv);

    // Read the hits from the relevant event file
    traccc::spacepoint_collection_types::host spacepoints_per_event(&resource);
    traccc::io::read_spacepoints(spacepoints_per_event, 0,
                                 "tml_full/single_muon/", &dd,
                                 traccc::data_format::csv);

    // Read the measurements from the relevant event file
    traccc::measurement_collection_types::host measurements_per_event(
        &resource);
    traccc::io::read_measurements(measurements_per_event, 0,
                                  "tml_full/single_muon/", &dd,
                                  traccc::data_format::csv);

    // Read the particles from the relevant event file
    traccc::particle_collection_types::host particles_per_event(&resource);
    traccc::io::read_particles(particles_per_event, 0, "tml_full/single_muon/",
                               traccc::data_format::csv);

    EXPECT_EQ(spacepoints_per_event.size(), 11u);
    EXPECT_EQ(measurements_per_event.size(), 11u);

    EXPECT_EQ(particles_per_event.size(), 1u);
}

/// Tests with ODD "single" muon events.
TEST_F(io, csv_read_odd_single_muon) {

    // Memory resource used by the test.
    vecmem::host_memory_resource mr;

    // Read the truth particles for the first event.
    traccc::particle_container_types::host particles{&mr};
    traccc::io::read_particles(particles, 0u, "odd/geant4_1muon_1GeV/",
                               traccc::data_format::csv);

    // Look at the read container.
    ASSERT_EQ(particles.size(), 265u);
    std::size_t n_muons = 0u;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        // The muon(s) must have measurements associated to it/them.
        if (std::abs(particles.at(i).header.particle_type) == 13) {
            ++n_muons;
            EXPECT_GT(particles.at(i).items.size(), 0u);
        }
    }
    EXPECT_EQ(n_muons, 4u);
}
