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
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/pixel_segmentation.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

TEST(algorithms, seq_single_module) {

    // Memory resource used in the test.
    vecmem::host_memory_resource resource;

    /// Following [DOI: 10.1109/DASIP48288.2019.9049184]
    traccc::host_cell_collection cells = {{{1, 0, 1., 0.},
                                           {8, 4, 2., 0.},
                                           {10, 4, 3., 0.},
                                           {9, 5, 4., 0.},
                                           {10, 5, 5., 0},
                                           {12, 12, 6, 0},
                                           {3, 13, 7, 0},
                                           {11, 13, 8, 0},
                                           {4, 14, 9, 0}},
                                          &resource};

    traccc::cell_module module;
    module.module = 0;
    module.pixel = traccc::pixel_segmentation{0., 0., 1., 1.};
    module.placement = traccc::transform3{};

    traccc::cluster_collection clusters;
    clusters.position_from_cell = module.pixel;

    traccc::host_measurement_collection measurements;

    traccc::host_spacepoint_collection spacepoints;

    traccc::component_connection cc(resource);
    traccc::measurement_creation mt(resource);
    traccc::spacepoint_formation sp(resource);

    // Algorithmic code: start
    clusters = cc(cells, module);
    measurements = mt(clusters, module);
    spacepoints = sp(module, measurements);
}
