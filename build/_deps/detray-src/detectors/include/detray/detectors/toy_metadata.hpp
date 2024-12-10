/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
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
#include "detray/geometry/shapes/concentric_cylinder2D.hpp"
#include "detray/geometry/shapes/rectangle2D.hpp"
#include "detray/geometry/shapes/ring2D.hpp"
#include "detray/geometry/shapes/trapezoid2D.hpp"
#include "detray/materials/material_map.hpp"
#include "detray/materials/material_slab.hpp"
#include "detray/navigation/accelerators/brute_force_finder.hpp"
#include "detray/navigation/accelerators/surface_grid.hpp"

namespace detray {

/// Defines the data types needed for the toy detector
struct toy_metadata {

    /// Define the algebra type for the geometry and navigation
    using algebra_type = ALGEBRA_PLUGIN<detray::scalar>;

    /// Mask to (next) volume link: next volume(s)
    using nav_link = std::uint_least16_t;

    /// Mask types
    using rectangle = mask<rectangle2D, nav_link>;
    using trapezoid = mask<trapezoid2D, nav_link>;
    // using cylinder = mask<cylinder2D, nav_link>;  // beampipe
    using cylinder_portal = mask<concentric_cylinder2D, nav_link>;
    using disc_portal = mask<ring2D, nav_link>;

    /// Material types
    using slab = material_slab<detray::scalar>;

    // Cylindrical material grid
    template <typename container_t>
    using cylinder_map_t =
        material_map<concentric_cylinder2D, scalar, container_t>;

    // Disc material grid
    template <typename container_t>
    using disc_map_t = material_map<ring2D, scalar, container_t>;

    // Rectangular material grid
    template <typename container_t>
    using rectangular_map_t = material_map<rectangle2D, scalar, container_t>;

    /// Surface grid types (regular, open binning)
    /// @{

    // Surface grid definition: bin-content: std::array<sf_descriptor, 1>
    template <typename axes_t, typename bin_entry_t, typename container_t>
    using surface_grid_t = grid<axes_t, bins::static_array<bin_entry_t, 1>,
                                simple_serializer, container_t, false>;

    // cylindrical grid for the barrel layers
    template <typename bin_entry_t, typename container_t>
    using cylinder_sf_grid =
        surface_grid_t<axes<concentric_cylinder2D>, bin_entry_t, container_t>;

    // disc grid for the endcap layers
    template <typename bin_entry_t, typename container_t>
    using disc_sf_grid = surface_grid_t<axes<ring2D>, bin_entry_t, container_t>;

    /// @}

    /// How to store coordinate transform matrices
    template <template <typename...> class vector_t = dvector>
    using transform_store =
        single_store<dtransform3D<algebra_type>, vector_t, geometry_context>;

    /// Mask type ids
    enum class mask_ids : std::uint_least8_t {
        e_rectangle2 = 0,
        e_trapezoid2 = 1,
        e_portal_cylinder2 = 2,
        e_portal_ring2 = 3,
        e_cylinder2 = 2,
    };

    /// How to store masks
    template <template <typename...> class tuple_t = dtuple,
              template <typename...> class vector_t = dvector>
    using mask_store =
        regular_multi_store<mask_ids, empty_context, tuple_t, vector_t,
                            rectangle, trapezoid, cylinder_portal, disc_portal>;

    /// Material type ids
    enum class material_ids : std::uint_least8_t {
        e_disc2_map = 0u,
        e_concentric_cylinder2_map = 1u,
        e_rectangle2_map = 2u,
        e_slab = 3u,
        e_none = 4u,
    };

    /// How to store materials
    template <template <typename...> class tuple_t = dtuple,
              typename container_t = host_container_types>
    using material_store =
        multi_store<material_ids, empty_context, tuple_t,
                    grid_collection<disc_map_t<container_t>>,
                    grid_collection<cylinder_map_t<container_t>>,
                    grid_collection<rectangular_map_t<container_t>>,
                    typename container_t::template vector_type<slab>>;

    /// How to link to the entries in the data stores
    using transform_link = typename transform_store<>::link_type;
    using mask_link = typename mask_store<>::single_link;
    using material_link = typename material_store<>::single_link;
    /// Surface type used for sensitives, passives and portals
    using surface_type =
        surface_descriptor<mask_link, material_link, transform_link, nav_link>;

    /// Portals and passives in the brute froce search, sensitives in the grids
    enum geo_objects : std::uint_least8_t {
        e_portal = 0,
        e_passive = 0,
        e_sensitive = 1,
        e_size = 2,
        e_all = e_size,
    };

    /// Acceleration data structures
    enum class accel_ids : std::uint_least8_t {
        e_brute_force = 0,     // test all surfaces in a volume (brute force)
        e_disc_grid = 1,       // endcap
        e_cylinder2_grid = 2,  // barrel
        e_default = e_brute_force,
    };

    /// One link for portals/passives and one sensitive surfaces
    using object_link_type =
        dmulti_index<dtyped_index<accel_ids, dindex>, geo_objects::e_size>;

    /// How to store the acceleration data structures
    template <template <typename...> class tuple_t = dtuple,
              typename container_t = host_container_types>
    using accelerator_store = multi_store<
        accel_ids, empty_context, tuple_t,
        brute_force_collection<surface_type, container_t>,
        grid_collection<disc_sf_grid<surface_type, container_t>>,
        grid_collection<cylinder_sf_grid<surface_type, container_t>>>;

    /// Volume search grid
    template <typename container_t = host_container_types>
    using volume_finder =
        grid<axes<cylinder3D, axis::bounds::e_open, axis::irregular,
                  axis::regular, axis::irregular>,
             bins::single<dindex>, simple_serializer, container_t>;
};

}  // namespace detray
