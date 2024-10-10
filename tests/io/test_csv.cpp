/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/details/read_surfaces.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector.hpp"
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

    traccc::edm::silicon_cell_collection::host cells{resource};
    traccc::io::read_cells(cells, get_datafile("single_module/cells.csv"));

    ASSERT_EQ(cells.size(), 6u);

    EXPECT_EQ(cells.channel0().at(0), 123u);
    EXPECT_EQ(cells.channel1().at(0), 32u);
    EXPECT_FLOAT_EQ(static_cast<float>(cells.activation().at(0)), 0.002f);
    EXPECT_EQ(cells.module_index().at(0), 0u);

    EXPECT_EQ(cells.channel0().at(5), 174u);
    EXPECT_EQ(cells.channel1().at(5), 880u);
    EXPECT_FLOAT_EQ(static_cast<float>(cells.activation().at(5)), 0.23f);
    EXPECT_EQ(cells.module_index().at(5), 0u);
}

// This defines the local frame test suite
TEST_F(io, csv_read_two_modules) {

    vecmem::host_memory_resource resource;

    traccc::edm::silicon_cell_collection::host cells{resource};
    traccc::io::read_cells(cells, get_datafile("two_modules/cells.csv"));

    ASSERT_EQ(cells.size(), 14u);

    // Check cells in first module
    EXPECT_EQ(cells.channel0().at(0), 123u);
    EXPECT_EQ(cells.channel1().at(0), 32u);
    EXPECT_EQ(cells.channel0().at(5), 174u);
    EXPECT_EQ(cells.channel1().at(5), 880u);

    // Check cells in second module
    EXPECT_EQ(cells.channel0().at(6), 0u);
    EXPECT_EQ(cells.channel1().at(6), 4u);
    EXPECT_EQ(cells.channel0().at(13), 5u);
    EXPECT_EQ(cells.channel1().at(13), 98u);
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

    traccc::edm::silicon_cell_collection::host cells{resource};
    traccc::io::read_cells(
        cells, get_datafile("tml_pixel_barrel/event000000000-cells.csv"),
        nullptr, traccc::data_format::csv, false);

    EXPECT_EQ(cells.size(), 179961u);
}

// This checks if hit and measurement container from the first single muon event
TEST_F(io, csv_read_tml_single_muon) {
    vecmem::host_memory_resource resource;

    // Read the detector description.
    traccc::silicon_detector_description::host dd{resource};
    traccc::io::read_detector_description(
        dd, "tml_detector/trackml-detector.csv",
        "tml_detector/default-geometric-config-generic.json",
        traccc::data_format::csv);

    // Read the hits from the relevant event file
    traccc::spacepoint_collection_types::host spacepoints_per_event(&resource);
    traccc::io::read_spacepoints(spacepoints_per_event, 0,
                                 "tml_full/single_muon/");

    // Read the measurements from the relevant event file
    traccc::measurement_collection_types::host measurements_per_event(
        &resource);
    traccc::io::read_measurements(measurements_per_event, 0,
                                  "tml_full/single_muon/", false, nullptr);

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

    traccc::default_detector::host detector{mr};
    traccc::io::read_detector(detector, mr,
                              "geometries/odd/odd-detray_geometry_detray.json");

    // Read the truth particles for the first event.
    traccc::particle_container_types::host particles{&mr};
    traccc::io::read_particles(particles, 0u, "odd/geant4_1muon_1GeV/", true,
                               &detector, traccc::data_format::csv);

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
