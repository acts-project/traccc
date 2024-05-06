/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
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
#include "traccc/io/read_spacepoints.hpp"

// Test include(s).
#include "tests/data_test.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

class io : public traccc::tests::data_test {};

// This reads in the tml pixel barrel first event
TEST_F(io, csv_read_tml_transforms) {
    std::string file = get_datafile("tml_detector/trackml-detector.csv");

    auto tml_barrel_transforms = traccc::io::details::read_surfaces(file);

    ASSERT_EQ(tml_barrel_transforms.size(), 18791u);
}

// This checks if hit and measurement container from the first single muon event
TEST_F(io, csv_read_tml_single_muon) {
    vecmem::host_memory_resource resource;

    // Read the surface transforms
    auto [surface_transforms, _] =
        traccc::io::read_geometry("tml_detector/trackml-detector.csv");

    // Read the hits from the relevant event file
    traccc::io::spacepoint_reader_output spacepoints_per_event(&resource);
    traccc::io::read_spacepoints(spacepoints_per_event, 0,
                                 "tml_full/single_muon/", surface_transforms,
                                 nullptr, traccc::data_format::csv);

    // Read the measurements from the relevant event file
    traccc::io::measurement_reader_output measurements_per_event(&resource);
    traccc::io::read_measurements(measurements_per_event, 0,
                                  "tml_full/single_muon/",
                                  traccc::data_format::csv);

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