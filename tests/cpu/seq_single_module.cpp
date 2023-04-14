/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/component_connection.hpp"
#include "traccc/clusterization/measurement_creation.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/edm/alt_measurement.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/pixel_data.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

TEST(algorithms, seq_single_module) {

    // Memory resource used in the test.
    vecmem::host_memory_resource resource;

    traccc::component_connection cc(resource);
    traccc::measurement_creation mc(resource);

    /// Following [DOI: 10.1109/DASIP48288.2019.9049184]
    traccc::cell_collection_types::host cells = {{{1, 0, 1., 0., 0},
                                                  {8, 4, 2., 0., 0},
                                                  {10, 4, 3., 0., 0},
                                                  {9, 5, 4., 0., 0},
                                                  {10, 5, 5., 0, 0},
                                                  {12, 12, 6, 0, 0},
                                                  {3, 13, 7, 0, 0},
                                                  {11, 13, 8, 0, 0},
                                                  {4, 14, 9, 0, 0}},
                                                 &resource};
    traccc::cell_module module;
    traccc::cell_module_collection_types::host modules(&resource);
    modules.push_back(module);

    auto clusters = cc(cells);
    EXPECT_EQ(clusters.size(), 4u);

    auto measurements = mc(clusters, modules);

    EXPECT_EQ(measurements.size(), 4u);
}
