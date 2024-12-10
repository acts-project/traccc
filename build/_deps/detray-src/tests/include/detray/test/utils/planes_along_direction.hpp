/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray core include(s).
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/containers.hpp"
#include "detray/geometry/detail/surface_descriptor.hpp"
#include "detray/utils/ranges.hpp"

// Detray tests include(s).
#include "detray/test/utils/types.hpp"

namespace detray::test {

enum class plane_mask_ids : unsigned int {
    e_plane_rectangle2 = 0u,
};

enum class plane_material_ids : unsigned int {
    e_plane_slab = 0u,
};

// Helper type definitions.
using plane_mask_link_t = dtyped_index<plane_mask_ids, dindex>;
using plane_material_link_t = dtyped_index<plane_material_ids, dindex>;

/// This method creates a number (distances.size()) planes along a direction
template <typename algebra_t = test::algebra>
auto planes_along_direction(const dvector<dscalar<algebra_t>> &distances,
                            const dvector3D<algebra_t> &direction) {

    using vector3_t = dvector3D<algebra_t>;
    using transform3_t = dtransform3D<algebra_t>;

    // New z- and x-axes
    vector3_t z{vector::normalize(direction)};
    vector3_t x = vector::normalize(vector3_t{0.f, -z[2], z[1]});

    dvector<surface_descriptor<plane_mask_link_t, plane_material_link_t>>
        surfaces;
    dvector<transform3_t> transforms;

    surfaces.reserve(distances.size());
    transforms.reserve(distances.size());

    for (const auto [idx, d] : detray::views::enumerate(distances)) {

        vector3_t t = d * direction;
        transforms.emplace_back(t, z, x);

        plane_mask_link_t mask_link{plane_mask_ids::e_plane_rectangle2, idx};
        plane_material_link_t material_link{plane_material_ids::e_plane_slab,
                                            0u};
        surfaces.emplace_back(idx, std::move(mask_link),
                              std::move(material_link), 0u,
                              surface_id::e_sensitive);
        surfaces.back().set_index(idx);
    }

    return std::make_tuple(std::move(surfaces), std::move(transforms));
}

}  // namespace detray::test
