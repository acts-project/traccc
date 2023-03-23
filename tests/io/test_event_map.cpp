/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/event_map2.hpp"

// GTest include(s).
#include <gtest/gtest.h>

// Test generate_particle_map function
TEST(event_map2, event_map2) {

    const std::string path =
        "detray_simulation/telescope/kf_validation/1_GeV_0_phi/";
    // Event map
    traccc::event_map2 evt_map(0, path, path, path);

    const auto& ptc_meas_map = evt_map.ptc_meas_map;
    const auto& meas_ptc_map = evt_map.meas_ptc_map;

    // Each particle makes 9 measurements in the telescope geometry
    for (auto const& [ptc, measurements] : ptc_meas_map) {
        ASSERT_EQ(measurements.size(), 9u);
    }

    // There is only one contributing particle for the measurement
    for (auto const& [measurements, ptc] : meas_ptc_map) {
        ASSERT_EQ(ptc.size(), 1u);
    }
}