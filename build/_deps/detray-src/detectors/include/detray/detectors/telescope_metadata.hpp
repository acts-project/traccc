/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/core/detail/multi_store.hpp"
#include "detray/core/detail/single_store.hpp"
#include "detray/definitions/detail/containers.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/geometry/detail/surface_descriptor.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes.hpp"
#include "detray/materials/material_rod.hpp"
#include "detray/materials/material_slab.hpp"
#include "detray/navigation/accelerators/brute_force_finder.hpp"

namespace detray {

/// Defines a telescope detector type with only rectangle portals and one
/// additional kind of contained module surfaces (@tparam mask_shape_t)
template <typename mask_shape_t = rectangle2D>
struct telescope_metadata {

    /// Define the algebra type for the geometry and navigation
    using algebra_type = ALGEBRA_PLUGIN<detray::scalar>;

    /// Mask to (next) volume link: next volume(s)
    using nav_link = std::uint_least16_t;

    /// Mask types (these types are needed for the portals, which are always
    /// there, and to resolve the wire surface material, i.e. slab vs. rod)
    using rectangle = mask<rectangle2D, nav_link>;
    using straw_tube = mask<line_circular, nav_link>;
    using drift_cell = mask<line_square, nav_link>;

    /// Material types
    using rod = material_rod<detray::scalar>;
    using slab = material_slab<detray::scalar>;

    /// How to store coordinate transform matrices
    template <template <typename...> class vector_t = dvector>
    using transform_store =
        single_store<dtransform3D<algebra_type>, vector_t, geometry_context>;

    /// Rectangles are always needed as portals (but the yhave the same type as
    /// module rectangles). Only one additional mask shape is allowed
    enum class mask_ids : std::uint_least8_t {
        e_rectangle2 = 0,
        e_portal_rectangle2 = 0,
        e_annulus2 = 1,
        e_cylinder2 = 1,
        e_ring2 = 1,
        e_trapezoid2 = 1,
        e_single1 = 1,
        e_single2 = 1,
        e_single3 = 1,
        e_straw_tube = 1,
        e_drift_cell = 1,
        e_unbounded_annulus2 = 1,
        e_unbounded_cell2 = 1,
        e_unbounded_cylinder2 = 1,
        e_unbounded_disc2 = 1,
        e_unbounded_rectangle2 = 1,
        e_unbounded_trapezoid2 = 1,
        e_unbounded_line_circular2 = 1,
        e_unmasked2 = 1,
    };

    /// How to store masks
    template <template <typename...> class tuple_t = dtuple,
              template <typename...> class vector_t = dvector>
    using mask_store = std::conditional_t<
        std::is_same_v<mask<mask_shape_t, nav_link>, rectangle>,
        regular_multi_store<mask_ids, empty_context, tuple_t, vector_t,
                            rectangle>,
        regular_multi_store<mask_ids, empty_context, tuple_t, vector_t,
                            rectangle, mask<mask_shape_t, nav_link>>>;

    /// Material type ids
    enum class material_ids : std::uint_least8_t {
        e_slab = 0u,
        e_raw_material = 1u,  //< used for homogeneous volume material
        e_rod = 2u,
        e_none = 3u,
    };

    /// How to store materials
    template <template <typename...> class tuple_t = dtuple,
              typename container_t = host_container_types>
    using material_store = std::conditional_t<
        std::is_same_v<mask<mask_shape_t, nav_link>, drift_cell> |
            std::is_same_v<mask<mask_shape_t, nav_link>, straw_tube>,
        regular_multi_store<material_ids, empty_context, tuple_t,
                            container_t::template vector_type, slab,
                            material<detray::scalar>, rod>,
        regular_multi_store<material_ids, empty_context, tuple_t,
                            container_t::template vector_type, slab,
                            material<detray::scalar>>>;

    /// How to link to the entries in the data stores
    using transform_link = typename transform_store<>::link_type;
    using mask_link = typename mask_store<>::single_link;
    using material_link = typename material_store<>::single_link;
    /// Surface type used for sensitives, passives and portals
    using surface_type =
        surface_descriptor<mask_link, material_link, transform_link, nav_link>;

    /// No grids/other acceleration data structure, everything is brute forced
    enum geo_objects : std::uint_least8_t {
        e_portal = 0,
        e_sensitive = 1,
        e_size = 2,
        e_all = e_size,
    };

    /// Acceleration data structures
    enum class accel_ids {
        e_brute_force = 0,  // test all surfaces in a volume (brute force)
        e_default = e_brute_force,
    };

    /// One link for all surfaces (in the brute force method)
    using object_link_type =
        dmulti_index<dtyped_index<accel_ids, dindex>, geo_objects::e_size>;

    /// How to store the brute force search data structure
    template <template <typename...> class tuple_t = dtuple,
              typename container_t = host_container_types>
    using accelerator_store =
        multi_store<accel_ids, empty_context, tuple_t,
                    brute_force_collection<surface_type, container_t>>;

    /// Volume search (only one volume exists)
    template <typename container_t = host_container_types>
    using volume_finder = brute_force_collection<dindex, container_t>;
};

}  // namespace detray
