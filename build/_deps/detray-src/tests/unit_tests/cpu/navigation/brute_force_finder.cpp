/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Detray include(s)
#include "detray/navigation/accelerators/brute_force_finder.hpp"

#include "detray/navigation/detail/ray.hpp"

// Detray test include(s)
#include "detray/test/utils/detectors/build_toy_detector.hpp"
#include "detray/test/utils/planes_along_direction.hpp"
#include "detray/test/utils/types.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s)
#include <gtest/gtest.h>

using namespace detray;

namespace {

vecmem::host_memory_resource host_mr;

// Algebra definitions
using vector3 = test::vector3;

/// A functor that performs some tests on a neighborhood of surfaces in a volume
struct neighbor_visit_test {

    /// Test the volume links
    template <typename surfaces_descriptor_t>
    DETRAY_HOST_DEVICE void operator()(const surfaces_descriptor_t& sf,
                                       const dindex test_vol_idx) const {
        EXPECT_EQ(sf.volume(), test_vol_idx)
            << " surface barcode: " << sf.barcode();
    }
};

}  // anonymous namespace

/// Test retrieval of surface from collection using brute force searching
GTEST_TEST(detray_navigation, brute_force_collection) {

    // Where to place the surfaces
    dvector<scalar> distances1{0.f, 10.0f, 20.0f, 40.0f, 80.0f, 100.0f};
    dvector<scalar> distances2{30.0f, 230.0f, 240.0f, 250.0f};
    dvector<scalar> distances3{0.1f, 5.0f, 50.0f, 500.0f, 5000.0f, 50000.0f};
    // surface direction
    vector3 direction{0.f, 0.f, 1.f};

    auto [surfaces1, transforms1] =
        test::planes_along_direction(distances1, direction);
    auto [surfaces2, transforms2] =
        test::planes_along_direction(distances2, direction);
    auto [surfaces3, transforms3] =
        test::planes_along_direction(distances3, direction);

    brute_force_collection<typename decltype(surfaces1)::value_type>
        sf_collection(&host_mr);

    // Check a few basics
    ASSERT_TRUE(sf_collection.empty());

    sf_collection.push_back(surfaces1);
    EXPECT_EQ(sf_collection.size(), 1UL);
    sf_collection.push_back(surfaces2);
    EXPECT_EQ(sf_collection.size(), 2UL);
    sf_collection.push_back(surfaces3);
    EXPECT_EQ(sf_collection.size(), 3UL);

    ASSERT_FALSE(sf_collection.empty());
    ASSERT_EQ(sf_collection.all().size(),
              distances1.size() + distances2.size() + distances3.size());

    // Check a single brute force finder
    EXPECT_EQ(sf_collection[0].size(), distances1.size());
    EXPECT_EQ(sf_collection[1].size(), distances2.size());
    EXPECT_EQ(sf_collection[2].size(), distances3.size());

    // Check the 'all' interface
    EXPECT_EQ(sf_collection[0].all().size(), distances1.size());
    EXPECT_EQ(sf_collection[1].all().size(), distances2.size());
    EXPECT_EQ(sf_collection[2].all().size(), distances3.size());

    // Check the test surfaces
    for (const auto& sf : sf_collection[1].all()) {
        EXPECT_EQ(sf.volume(), 0UL);
        EXPECT_EQ(sf.id(), surface_id::e_sensitive);
        EXPECT_FALSE(sf.is_portal());
    }
}

/// Integration test for the retrieval of surfaces in a volume during local
/// navigation
GTEST_TEST(detray_navigation, brute_force_search) {

    const auto [det, names] = build_toy_detector(host_mr);

    using detector_t = decltype(det);
    using context_t = detector_t::geometry_context;
    context_t ctx{};

    struct navigation_cfg {
        std::array<dindex, 2> search_window;
    };

    // Now run a brute force surface search in the first barrel layer
    dindex test_vol_idx{7UL};
    const auto vol = tracking_volume{det, test_vol_idx};

    // track in x-direction
    detail::ray<typename detector_t::algebra_type> trk({0.f, 0.f, 0.f}, 0.f,
                                                       {1.f, 0.f, 0.f}, -1.f);

    vol.template visit_neighborhood<neighbor_visit_test>(trk, navigation_cfg{},
                                                         ctx, test_vol_idx);
}
