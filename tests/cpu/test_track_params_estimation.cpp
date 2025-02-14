/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// Detray include(s).
#include <detray/tracks/helix.hpp>

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

using namespace traccc;

namespace {

// Memory resource used by the EDM.
vecmem::host_memory_resource host_mr;

/// Helper function for creating a spacepoint
void add_spacepoint(measurement_collection_types::host& measurements,
                    edm::spacepoint_collection::host& spacepoints,
                    const point3& pos) {

    const auto i = spacepoints.size();
    spacepoints.resize(i + 1);
    auto sp = spacepoints.at(i);
    sp.x() = pos[0];
    sp.y() = pos[1];
    sp.z() = pos[2];
    sp.measurement_index() = static_cast<unsigned int>(measurements.size());
    measurements.push_back({});
}

}  // namespace

TEST(track_params_estimation, helix_negative_charge) {

    // Set B field
    const vector3 B{0.f * unit<scalar>::T, 0.f * unit<scalar>::T,
                    2.f * unit<scalar>::T};

    // Track property
    const scalar q{-1.f * unit<scalar>::e};
    const point3 pos{0.f, 0.f, 0.f};
    const scalar time{0.f};
    const vector3 mom{1.f * unit<scalar>::GeV, 0.f, 1.f * unit<scalar>::GeV};

    // Make a helix
    detray::detail::helix<traccc::default_algebra> hlx(
        pos, time, vector::normalize(mom), q / vector::norm(mom), &B);

    // Make three spacepoints with the helix
    measurement_collection_types::host measurements(&host_mr);
    edm::spacepoint_collection::host spacepoints{host_mr};
    add_spacepoint(measurements, spacepoints, hlx(50 * unit<scalar>::mm));
    add_spacepoint(measurements, spacepoints, hlx(100 * unit<scalar>::mm));
    add_spacepoint(measurements, spacepoints, hlx(150 * unit<scalar>::mm));

    // Make a seed from the three spacepoints
    edm::seed_collection::host seeds{host_mr};
    seeds.resize(1);
    auto seed = seeds.at(0);
    seed.bottom_index() = 0u;
    seed.middle_index() = 1u;
    seed.top_index() = 2u;

    // Run track parameter estimation
    traccc::host::track_params_estimation tp(host_mr);
    auto bound_params =
        tp(vecmem::get_data(measurements), vecmem::get_data(spacepoints),
           vecmem::get_data(seeds), B);

    // Make sure that the reconstructed momentum is equal to the original
    // momentum
    ASSERT_EQ(bound_params.size(), 1u);
    ASSERT_NEAR(bound_params[0].p(q), vector::norm(mom), 2.f * 1e-4);
    ASSERT_TRUE(bound_params[0].qop() < 0.f);
}

TEST(track_params_estimation, helix_positive_charge) {

    // Set B field
    const vector3 B{0.f * unit<scalar>::T, 0.f * unit<scalar>::T,
                    2.f * unit<scalar>::T};

    // Track property
    const scalar q{1.f * unit<scalar>::e};
    const point3 pos{0.f, 0.f, 0.f};
    const scalar time{0.f};
    const vector3 mom{1.f * unit<scalar>::GeV, 0.f, 1.f * unit<scalar>::GeV};

    // Make a helix
    detray::detail::helix<traccc::default_algebra> hlx(
        pos, time, vector::normalize(mom), q / vector::norm(mom), &B);

    // Make three spacepoints with the helix
    measurement_collection_types::host measurements(&host_mr);
    edm::spacepoint_collection::host spacepoints{host_mr};
    add_spacepoint(measurements, spacepoints, hlx(50 * unit<scalar>::mm));
    add_spacepoint(measurements, spacepoints, hlx(100 * unit<scalar>::mm));
    add_spacepoint(measurements, spacepoints, hlx(150 * unit<scalar>::mm));

    // Make a seed from the three spacepoints
    edm::seed_collection::host seeds{host_mr};
    seeds.resize(1);
    auto seed = seeds.at(0);
    seed.bottom_index() = 0u;
    seed.middle_index() = 1u;
    seed.top_index() = 2u;

    // Run track parameter estimation
    traccc::host::track_params_estimation tp(host_mr);
    auto bound_params =
        tp(vecmem::get_data(measurements), vecmem::get_data(spacepoints),
           vecmem::get_data(seeds), B);

    // Make sure that the reconstructed momentum is equal to the original
    // momentum
    ASSERT_EQ(bound_params.size(), 1u);
    ASSERT_NEAR(bound_params[0].p(q), vector::norm(mom), 2.f * 1e-4);
    ASSERT_TRUE(bound_params[0].qop() > 0.f);
}
