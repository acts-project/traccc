/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/core/detail/multi_store.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// Google Test include(s)
#include <gtest/gtest.h>

// This tests the construction of a static transform store
GTEST_TEST(detray_core, static_mask_store) {

    vecmem::host_memory_resource host_mr;

    using namespace detray;

    enum class mask_ids : unsigned int {
        e_rectangle2 = 0u,
        e_trapezoid2 = 1u,
        e_annulus2 = 2u,
        e_cylinder3 = 3u,
        e_ring2 = 4u,
        e_single3 = 5u,
    };

    using rectangle = mask<rectangle2D>;
    using trapezoid = mask<trapezoid2D>;
    using annulus = mask<annulus2D>;
    using cylinder = mask<cylinder2D>;
    using ring = mask<ring2D>;
    using single = mask<single3D<>>;

    // Types must be sorted according to their id (here: masks/mask_identifier)
    using mask_container_t =
        regular_multi_store<mask_ids, empty_context, dtuple, dvector, rectangle,
                            trapezoid, annulus, cylinder, ring, single>;

    mask_container_t store(host_mr);

    ASSERT_TRUE(store.empty<mask_ids::e_annulus2>());
    ASSERT_TRUE(store.empty<mask_ids::e_cylinder3>());
    ASSERT_TRUE(store.empty<mask_ids::e_rectangle2>());
    ASSERT_TRUE(store.empty<mask_ids::e_ring2>());
    ASSERT_TRUE(store.empty<mask_ids::e_single3>());
    ASSERT_TRUE(store.empty<mask_ids::e_trapezoid2>());

    store.emplace_back<mask_ids::e_cylinder3>(empty_context{}, 0u, 1.f, 0.5f,
                                              2.0f);

    ASSERT_TRUE(store.empty<mask_ids::e_annulus2>());
    ASSERT_EQ(store.size<mask_ids::e_cylinder3>(), 1);
    ASSERT_TRUE(store.empty<mask_ids::e_rectangle2>());
    ASSERT_TRUE(store.empty<mask_ids::e_ring2>());
    ASSERT_TRUE(store.empty<mask_ids::e_single3>());
    ASSERT_TRUE(store.empty<mask_ids::e_trapezoid2>());

    store.emplace_back<mask_ids::e_cylinder3>(empty_context{}, 0u, 1.f, 1.5f,
                                              2.0f);
    store.emplace_back<mask_ids::e_trapezoid2>(empty_context{}, 0u, 0.5f, 1.5f,
                                               4.0f, 1.f / 8.f);
    store.emplace_back<mask_ids::e_rectangle2>(empty_context{}, 0u, 1.0f, 2.0f);
    store.emplace_back<mask_ids::e_rectangle2>(empty_context{}, 0u, 2.0f, 1.0f);
    store.emplace_back<mask_ids::e_rectangle2>(empty_context{}, 0u, 10.0f,
                                               100.0f);

    ASSERT_TRUE(store.empty<mask_ids::e_annulus2>());
    ASSERT_EQ(store.size<mask_ids::e_cylinder3>(), 2);
    ASSERT_EQ(store.size<mask_ids::e_rectangle2>(), 3);
    ASSERT_TRUE(store.empty<mask_ids::e_ring2>());
    ASSERT_TRUE(store.empty<mask_ids::e_single3>());
    ASSERT_EQ(store.size<mask_ids::e_trapezoid2>(), 1);

    store.emplace_back<mask_ids::e_annulus2>(empty_context{}, 0u, 1.f, 2.f, 3.f,
                                             4.f, 5.f, 6.f, 7.f);
    store.emplace_back<mask_ids::e_ring2>(empty_context{}, 0u, 10.f, 100.f);
    store.emplace_back<mask_ids::e_ring2>(empty_context{}, 0u, 10.f, 100.f);
    store.emplace_back<mask_ids::e_ring2>(empty_context{}, 0u, 10.f, 100.f);
    store.emplace_back<mask_ids::e_ring2>(empty_context{}, 0u, 10.f, 100.f);

    const auto &annulus_masks = store.get<mask_ids::e_annulus2>();
    const auto &cylinder_masks = store.get<mask_ids::e_cylinder3>();
    const auto &rectangle_masks = store.get<mask_ids::e_rectangle2>();
    const auto &ring_masks = store.get<mask_ids::e_ring2>();
    const auto &single_masks = store.get<mask_ids::e_single3>();
    const auto &trapezoid_masks = store.get<mask_ids::e_trapezoid2>();

    ASSERT_TRUE(annulus_masks.size() == 1);
    ASSERT_TRUE(cylinder_masks.size() == 2);
    ASSERT_TRUE(rectangle_masks.size() == 3);
    ASSERT_TRUE(ring_masks.size() == 4);
    ASSERT_TRUE(single_masks.size() == 0);
    ASSERT_TRUE(trapezoid_masks.size() == 1);
}
