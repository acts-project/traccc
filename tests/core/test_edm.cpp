/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/edm/TrackContainer.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

TEST(edm, track_container_creation) {

    // Create an empty track container.
    vecmem::host_memory_resource mr;
    traccc::edm::track_container<traccc::default_algebra>::host tracks{mr};
    traccc::edm::track_container<traccc::default_algebra>::const_data
        tracksData{tracks};
    Acts::TrackingGeometry* dummyActsGeometry = nullptr;
    traccc::host_detector dummyDetrayGeometry;

    // Create the track container.
    traccc::edm::TrackContainer trackContainer{
        traccc::edm::TrackContainerBackend{tracksData, *dummyActsGeometry,
                                           dummyDetrayGeometry},
        traccc::edm::MultiTrajectory{tracksData, *dummyActsGeometry,
                                     dummyDetrayGeometry}};

    // Check that the size is zero.
    ASSERT_EQ(trackContainer.size(), 0u);
}
