/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

#include <iostream>

using namespace traccc;

// Seeding with two muons
TEST(seeding, two_low_pT_muons) {

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Config objects
    traccc::seedfinder_config finder_config;
    traccc::spacepoint_grid_config grid_config(finder_config);
    traccc::seedfilter_config filter_config;

    // Adjust parameters
    finder_config.deltaRMax = 100. * unit<scalar>::mm;
    finder_config.maxPtScattering = 0.5 * unit<scalar>::GeV;
    traccc::seeding_algorithm sa(finder_config, grid_config, filter_config,
                                 host_mr);
    spacepoint_collection_types::host spacepoints;

    // Spacepoints from 16.42 GeV muon
    spacepoints.push_back({{36.6706, 10.6472, 104.131}, {}});
    spacepoints.push_back({{94.2191, 29.6699, 113.628}, {}});
    spacepoints.push_back({{149.805, 47.9518, 122.979}, {}});
    spacepoints.push_back({{218.514, 70.3049, 134.029}, {}});
    spacepoints.push_back({{275.359, 88.668, 143.378}, {}});

    // Spacepoints from 1.8 GeV muon
    spacepoints.push_back({{36.301, 13.1197, 106.83}, {}});
    spacepoints.push_back({{93.9366, 33.7101, 120.978}, {}});
    spacepoints.push_back({{149.192, 52.0562, 134.678}, {}});
    spacepoints.push_back({{218.398, 73.1025, 151.979}, {}});
    spacepoints.push_back({{275.322, 89.0663, 166.229}, {}});

    // Run seeding
    auto seeds = sa(spacepoints);

    // The number of seeds should be eqaul to two
    ASSERT_EQ(seeds.size(), 2u);
}