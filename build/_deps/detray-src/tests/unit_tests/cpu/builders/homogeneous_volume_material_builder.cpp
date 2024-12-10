/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Detray include(s)
#include "detray/builders/homogeneous_volume_material_builder.hpp"

#include "detray/core/detector.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/materials/predefined_materials.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s)
#include <gtest/gtest.h>

// System include(s)
#include <limits>
#include <memory>
#include <vector>

using namespace detray;

/// Unittest: Test the construction of a collection of materials
TEST(detray_tools, homogeneous_volume_material_builder) {

    using detector_t = detector<>;

    constexpr auto material_id{detector_t::materials::id::e_raw_material};

    vecmem::host_memory_resource host_mr;
    detector_t d(host_mr);

    EXPECT_TRUE(d.material_store().template empty<material_id>());

    // Add material to a new volume
    auto vbuilder =
        std::make_unique<volume_builder<detector_t>>(volume_id::e_cylinder);
    auto mat_builder =
        homogeneous_volume_material_builder<detector_t>{std::move(vbuilder)};

    mat_builder.set_material(argon_liquid<scalar>{});

    // Add the volume to the detector
    mat_builder.build(d);

    // Test the material data
    EXPECT_EQ(d.volumes().size(), 1u);
    const auto &vol_desc = d.volumes().at(0);
    EXPECT_EQ(vol_desc.material().id(), material_id);
    EXPECT_EQ(vol_desc.material().index(), 0u);
    EXPECT_EQ(d.material_store().template size<material_id>(), 1u);
    EXPECT_EQ(d.material_store().template get<material_id>()[0],
              argon_liquid<scalar>{});
}
