/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/utils.hpp"

// Detray include(s)
#include "detray/core/detector.hpp"
#include "detray/io/common/detector_reader.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System
#include <set>

// This defines the local frame test suite for the digitization configuration
TEST(io_digi_configurator, toy_detector) {

    // Read the digitization configuration file
    auto digi_cfg = traccc::io::read_digitization_config(
        "geometries/toy_detector/toy_detector-geometric-config.json");

    // Get the detector as reference
    vecmem::host_memory_resource host_mr;
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(traccc::io::data_directory() +
                        "geometries/toy_detector/toy_detector_geometry.json");
    const auto [det, names] =
        detray::io::read_detector<detray::detector<>>(host_mr, reader_cfg);

    std::size_t n_sensitives{0u};
    std::set<std::size_t> volumes;
    for (const auto& sf_desc : det.surface_lookup()) {
        if (sf_desc.is_sensitive()) {
            volumes.insert(sf_desc.volume());
            ++n_sensitives;
        }
    }

    ASSERT_EQ(volumes.size(), 18u);
    ASSERT_EQ(n_sensitives, digi_cfg.size());
}
