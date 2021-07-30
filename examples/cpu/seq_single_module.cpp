/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <vecmem/memory/host_memory_resource.hpp>

#include "clusterization/component_connection.hpp"
#include "clusterization/measurement_creation.hpp"
#include "clusterization/spacepoint_formation.hpp"
#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"
#include "geometry/pixel_segmentation.hpp"

int main() {

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

    traccc::component_connection cc;
    traccc::measurement_creation mt;
    traccc::spacepoint_formation sp;

    // Algorithmic code: start
    clusters = cc({cells, module});
    measurements = mt({clusters, module});
    spacepoints = sp({module, measurements});

    return 0;
}
