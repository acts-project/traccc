/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/definitions/detail/indexing.hpp"

// Google test include(s)
#include <gtest/gtest.h>

using namespace detray;

namespace {

/// Define mask types
enum class mask_ids : unsigned int {
    e_unmasked = 0u,
};

}  // namespace

/// Test the typed index
GTEST_TEST(detray_core, typed_index) {

    using index_t = dtyped_index<mask_ids, unsigned int>;
    auto ti = index_t{};

    // Check a empty barcode
    EXPECT_EQ(ti.id(), static_cast<mask_ids>((1u << 4) - 1u));
    EXPECT_EQ(ti.index(), static_cast<unsigned int>((1u << 28) - 1u));

    ti.set_id(mask_ids::e_unmasked).set_index(42u);

    // Check the values after setting them
    EXPECT_EQ(ti.id(), mask_ids::e_unmasked);
    EXPECT_EQ(ti.index(), 42u);

    // Check invalid link
    EXPECT_FALSE(ti.is_invalid());
    ti.set_id(static_cast<index_t::id_type>((1u << 4) - 1u));
    EXPECT_TRUE(ti.is_invalid());
    ti.set_id(mask_ids::e_unmasked);
    EXPECT_FALSE(ti.is_invalid());
    ti.set_index((1u << 30) - 1u);
    EXPECT_TRUE(ti.is_invalid());
}
