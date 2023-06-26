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
#include "detray/intersection/detail/trajectories.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

using namespace traccc;

TEST(track_params_estimation, helix) {

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Set B field
    const vector3 B{0. * unit<scalar>::T, 0. * unit<scalar>::T,
                    2. * unit<scalar>::T};

    // Track property
    const point3 pos{0.f, 0.f, 0.f};
    const scalar time{0.f};
    const vector3 mom{1.f, 0.f, 1.f * unit<scalar>::GeV};
    const scalar q{-1.f * unit<scalar>::e};

    // Make a helix
    detray::detail::helix<transform3> hlx(pos, time, mom, q, &B);

    // Make three spacepoints with the helix
    spacepoint_collection_types::host spacepoints;
    spacepoints.push_back({hlx(50 * unit<scalar>::mm), {}});
    spacepoints.push_back({hlx(100 * unit<scalar>::mm), {}});
    spacepoints.push_back({hlx(150 * unit<scalar>::mm), {}});

    // Make a seed from the three spacepoints
    seed_collection_types::host seeds;
    seeds.push_back({0u, 1u, 2u, 0.f, 0.f});

    // Run track parameter estimation
    traccc::track_params_estimation tp(host_mr);
    auto bound_params = tp(spacepoints, seeds, B);

    // Make sure that the reconstructed momentum is equal to the original
    // momentum
    ASSERT_EQ(bound_params.size(), 1u);
    ASSERT_NEAR(bound_params[0].p(), getter::norm(mom), 1e-4);
}