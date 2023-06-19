/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// Detray include(s).
#include "detray/detectors/create_toy_geometry.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

using namespace traccc;

namespace {

// Memory resource used by the EDM.
vecmem::host_memory_resource host_mr;

// Set B field
static constexpr vector3 B{0. * unit<scalar>::T, 0. * unit<scalar>::T,
                           2. * unit<scalar>::T};

}  // namespace

// Seeding with two muons
TEST(seeding, case1) {

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

    // Spacepoints from 16.62 GeV muon
    spacepoints.push_back({{36.6706, 10.6472, 104.131}, {}});
    spacepoints.push_back({{94.2191, 29.6699, 113.628}, {}});
    spacepoints.push_back({{149.805, 47.9518, 122.979}, {}});
    spacepoints.push_back({{218.514, 70.3049, 134.029}, {}});
    spacepoints.push_back({{275.359, 88.668, 143.378}, {}});

    // Run seeding
    auto seeds = sa(spacepoints);

    // The number of seeds should be eqaul to one
    ASSERT_EQ(seeds.size(), 1u);

    traccc::track_params_estimation tp(host_mr);

    auto bound_params = tp(spacepoints, seeds, B);

    // The number of bound track parameters should be eqaul to one
    ASSERT_EQ(bound_params.size(), 1u);

    // Make sure that we have reasonable estimation on momentum
    /* Currently disabled
    EXPECT_NEAR(bound_params[0].p(), 16.62 * unit<scalar>::GeV,
                0.1 * unit<scalar>::GeV);
    */
}

TEST(seeding, case2) {

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

    // Spacepoints from 1.85 GeV muon
    spacepoints.push_back({{36.301, 13.1197, 106.83}, {}});
    spacepoints.push_back({{93.9366, 33.7101, 120.978}, {}});
    spacepoints.push_back({{149.192, 52.0562, 134.678}, {}});
    spacepoints.push_back({{218.398, 73.1025, 151.979}, {}});
    spacepoints.push_back({{275.322, 89.0663, 166.229}, {}});

    // Run seeding
    auto seeds = sa(spacepoints);

    // The number of seeds should be eqaul to one
    ASSERT_EQ(seeds.size(), 1u);

    traccc::track_params_estimation tp(host_mr);

    auto bound_params = tp(spacepoints, seeds, B);

    // The number of bound track parameters should be eqaul to one
    ASSERT_EQ(bound_params.size(), 1u);

    // Make sure that we have reasonable estimation on momentum
    /* Currently disabled
    EXPECT_NEAR(bound_params[0].p(), 1.85 * unit<scalar>::GeV,
                0.1 * unit<scalar>::GeV);
    */
}