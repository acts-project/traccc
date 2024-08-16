/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/measurement_creation_algorithm.hpp"
#include "traccc/clusterization/sparse_ccl_algorithm.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

TEST(algorithms, seq_single_module) {

    // Memory resource used in the test.
    vecmem::host_memory_resource resource;

    traccc::host::sparse_ccl_algorithm cc(resource);
    traccc::host::measurement_creation_algorithm mc(resource);

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
    traccc::silicon_detector_description::host dd{resource};
    dd.resize(1);

    auto clusters = cc(vecmem::get_data(cells));
    EXPECT_EQ(clusters.size(), 4u);

    auto clusters_data = traccc::get_data(clusters);
    auto dd_data = vecmem::get_data(dd);
    auto measurements = mc(clusters_data, dd_data);

    EXPECT_EQ(measurements.size(), 4u);
}
