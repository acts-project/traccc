/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/core/detector.hpp"

#include "detray/definitions/detail/indexing.hpp"
#include "detray/materials/predefined_materials.hpp"

// Detray test include(s)
#include "detray/test/utils/prefill_detector.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s)
#include <gtest/gtest.h>

/// This tests the functionality of a detector as a data store manager
GTEST_TEST(detray_core, detector) {

    using namespace detray;

    using detector_t = detector<>;
    using mask_id = typename detector_t::masks::id;
    using material_id = typename detector_t::materials::id;
    using finder_id = typename detector_t::accel::id;

    vecmem::host_memory_resource host_mr;
    detector_t d1(host_mr);
    auto geo_ctx = typename detector_t::geometry_context{};

    // Helper lambda for checking the contents of an "empty" detector object.
    auto check_empty_detector = [](auto& d) {
        EXPECT_TRUE(d.volumes().empty());
        EXPECT_TRUE(d.portals().empty());
        EXPECT_TRUE(d.transform_store().empty());
        EXPECT_TRUE(d.mask_store().template empty<mask_id::e_rectangle2>());
        EXPECT_TRUE(
            d.mask_store().template empty<mask_id::e_portal_rectangle2>());
        EXPECT_TRUE(d.mask_store().template empty<mask_id::e_trapezoid2>());
        EXPECT_TRUE(d.mask_store().template empty<mask_id::e_annulus2>());
        EXPECT_TRUE(d.mask_store().template empty<mask_id::e_cylinder2>());
        EXPECT_TRUE(
            d.mask_store().template empty<mask_id::e_portal_cylinder2>());
        EXPECT_TRUE(d.mask_store().template empty<mask_id::e_ring2>());
        EXPECT_TRUE(d.mask_store().template empty<mask_id::e_portal_ring2>());
        EXPECT_TRUE(d.mask_store().template empty<mask_id::e_straw_tube>());
        EXPECT_TRUE(d.mask_store().template empty<mask_id::e_drift_cell>());
        EXPECT_TRUE(d.material_store().template empty<material_id::e_slab>());
        EXPECT_TRUE(d.material_store().template empty<material_id::e_rod>());
        EXPECT_TRUE(
            d.accelerator_store().template empty<finder_id::e_brute_force>());
        EXPECT_TRUE(
            d.accelerator_store().template empty<finder_id::e_disc_grid>());
        EXPECT_TRUE(d.accelerator_store()
                        .template empty<finder_id::e_cylinder2_grid>());
        EXPECT_TRUE(
            d.accelerator_store().template empty<finder_id::e_irr_disc_grid>());
        EXPECT_TRUE(d.accelerator_store()
                        .template empty<finder_id::e_irr_cylinder2_grid>());
        EXPECT_TRUE(
            d.accelerator_store().template empty<finder_id::e_default>());
    };

    // Check the empty detector object.
    check_empty_detector(d1);

    // Add some geometrical data
    prefill_detector(d1, geo_ctx);

    // Helper lambda for checking the contents of a "filled" detector object.
    auto check_filled_detector = [](auto& d) {
        // TODO: add B-field check
        EXPECT_EQ(d.volumes().size(), 1u);
        EXPECT_EQ(d.portals().size(), 3u);
        EXPECT_EQ(d.transform_store().size(), 4u);
        EXPECT_EQ(d.mask_store().template size<mask_id::e_rectangle2>(), 1u);
        EXPECT_EQ(d.mask_store().template size<mask_id::e_portal_rectangle2>(),
                  1u);
        EXPECT_EQ(d.mask_store().template size<mask_id::e_trapezoid2>(), 1u);
        EXPECT_EQ(d.mask_store().template size<mask_id::e_annulus2>(), 1u);
        EXPECT_EQ(d.mask_store().template size<mask_id::e_cylinder2>(), 0u);
        EXPECT_EQ(d.mask_store().template size<mask_id::e_portal_cylinder2>(),
                  0u);
        EXPECT_EQ(d.mask_store().template size<mask_id::e_ring2>(), 0u);
        EXPECT_EQ(d.mask_store().template size<mask_id::e_portal_ring2>(), 0u);
        EXPECT_EQ(d.mask_store().template size<mask_id::e_straw_tube>(), 0u);
        EXPECT_EQ(d.mask_store().template size<mask_id::e_drift_cell>(), 0u);
        EXPECT_EQ(d.material_store().template size<material_id::e_slab>(), 2u);
        EXPECT_EQ(d.material_store().template size<material_id::e_rod>(), 1u);
        EXPECT_EQ(
            d.accelerator_store().template size<finder_id::e_brute_force>(),
            1u);
        EXPECT_EQ(d.accelerator_store().template size<finder_id::e_disc_grid>(),
                  0u);
        EXPECT_EQ(
            d.accelerator_store().template size<finder_id::e_cylinder2_grid>(),
            0u);
        EXPECT_EQ(
            d.accelerator_store().template size<finder_id::e_irr_disc_grid>(),
            0u);
        EXPECT_EQ(d.accelerator_store()
                      .template size<finder_id::e_irr_cylinder2_grid>(),
                  0u);
        EXPECT_EQ(d.accelerator_store().template size<finder_id::e_default>(),
                  1u);
    };

    // Check the filled detector object.
    check_filled_detector(d1);

    // Move construct a detector object.
    detector_t d2{std::move(d1)};
    check_filled_detector(d2);

    // Create a new, empty detector.
    detector_t d3{host_mr};
    check_empty_detector(d3);

    // Move assign the filled detector to the empty one.
    d3 = std::move(d2);
    check_filled_detector(d3);
}
