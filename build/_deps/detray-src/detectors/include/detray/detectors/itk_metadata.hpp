/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
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
#include "detray/geometry/shapes/annulus2D.hpp"
#include "detray/geometry/shapes/concentric_cylinder2D.hpp"
#include "detray/geometry/shapes/rectangle2D.hpp"
#include "detray/geometry/shapes/ring2D.hpp"
#include "detray/materials/material_map.hpp"
#include "detray/materials/material_slab.hpp"
#include "detray/navigation/accelerators/brute_force_finder.hpp"
#include "detray/navigation/accelerators/surface_grid.hpp"

// Linear algebra types
#include "detray/definitions/detail/algebra.hpp"

namespace detray {

//
// Detector
//

/// Defines a detector that contains squares, trapezoids and a bounding portal
/// box.
struct itk_metadata {

    /// Define the algebra type for the geometry and navigation
    using algebra_type = ALGEBRA_PLUGIN<detray::scalar>;

    /// Portal link type between volumes
    using nav_link = std::uint_least16_t;

    //
    // Surface Primitives
    //

    /// The mask types for the detector sensitive surfaces
    using annulus = mask<annulus2D, nav_link>;
    using rectangle = mask<rectangle2D, nav_link>;
    // Types for portals
    using cylinder_portal = mask<concentric_cylinder2D, nav_link>;
    using disc_portal = mask<ring2D, nav_link>;

    //
    // Material Description
    //

    /// The material types to be mapped onto the surfaces: Here homogeneous
    /// material
    using slab = material_slab<detray::scalar>;

    // Cylindrical material map
    template <typename container_t>
    using cylinder_map_t =
        material_map<concentric_cylinder2D, scalar, container_t>;

    // Disc material map
    template <typename container_t>
    using disc_map_t = material_map<ring2D, scalar, container_t>;

    // Rectangular material map
    template <typename container_t>
    using rectangular_map_t = material_map<rectangle2D, scalar, container_t>;

    /// How to store and link transforms. The geometry context allows to resolve
    /// the conditions data for e.g. module alignment
    template <template <typename...> class vector_t = dvector>
    using transform_store =
        single_store<dtransform3D<algebra_type>, vector_t, geometry_context>;

    /// Assign the mask types to the mask tuple container entries. It may be a
    /// good idea to have the most common types in the first tuple entries, in
    /// order to minimize the depth of the 'unrolling' before a mask is found
    /// in the tuple
    enum class mask_ids : std::uint_least8_t {
        e_rectangle2 = 0u,
        e_annulus2 = 1u,
        e_portal_cylinder2 = 2u,
        e_portal_ring2 = 3u,
    };

    /// This is the mask collections tuple (in the detector called 'mask store')
    /// the @c regular_multi_store is a vecemem-ready tuple of vectors of
    /// the detector masks.
    template <template <typename...> class tuple_t = dtuple,
              template <typename...> class vector_t = dvector>
    using mask_store =
        regular_multi_store<mask_ids, empty_context, tuple_t, vector_t,
                            rectangle, annulus, cylinder_portal, disc_portal>;

    /// Similar to the mask store, there is a material store, which
    enum class material_ids : std::uint_least8_t {
        e_disc2_map = 0u,
        e_annulus2_map = 0u,
        e_concentric_cylinder2_map = 1u,
        e_reactangle2_map = 2u,
        e_slab = 3u,  //< keep for the EF-tracking geometry
        e_none = 4u,
    };

    /// How to store and link materials. The material does not make use of
    /// conditions data ( @c empty_context )
    template <template <typename...> class tuple_t = dtuple,
              typename container_t = host_container_types>
    using material_store =
        multi_store<material_ids, empty_context, tuple_t,
                    grid_collection<disc_map_t<container_t>>,
                    grid_collection<cylinder_map_t<container_t>>,
                    grid_collection<rectangular_map_t<container_t>>,
                    typename container_t::template vector_type<slab>>;

    /// Surface descriptor type used for sensitives, passives and portals
    /// It holds the indices to the surface data in the detector data stores
    /// that were defined above
    using transform_link = typename transform_store<>::link_type;
    using mask_link = typename mask_store<>::single_link;
    using material_link = typename material_store<>::single_link;
    /// Surface type used for sensitives, passives and portals
    using surface_type =
        surface_descriptor<mask_link, material_link, transform_link, nav_link>;

    /// How to index the constituent objects in a volume
    /// If they share the same index value here, they will be added into the
    /// same acceleration data structure in every respective volume
    enum geo_objects : std::uint_least8_t {
        e_surface = 0u,  //< This detector keeps all surfaces in the same
        e_portal = 0u,   //  acceleration data structure (id 0)
        e_passive = 0u,
        e_size = 1u
    };

    /// The acceleration data structures live in another tuple that needs to be
    /// indexed correctly:
    enum class accel_ids : std::uint_least8_t {
        e_brute_force = 0u,  //< test all surfaces in a volume (brute force)
        e_default = e_brute_force,
    };

    /// How a volume finds its constituent objects in the detector containers
    /// In this case: One range for sensitive/passive surfaces, one for portals
    using object_link_type =
        dmulti_index<dtyped_index<accel_ids, dindex>, geo_objects::e_size>;

    /// The tuple store that hold the acceleration data structures for all
    /// volumes. Every collection of accelerationdata structures defines its
    /// own container and view type. Does not make use of conditions data
    /// ( @c empty_context )
    template <template <typename...> class tuple_t = dtuple,
              typename container_t = host_container_types>
    using accelerator_store =
        multi_store<accel_ids, empty_context, tuple_t,
                    brute_force_collection<surface_type, container_t>>;

    /// Data structure that allows to find the current detector volume from a
    /// given position. Here: Uniform grid with a 3D cylindrical shape
    template <typename container_t = host_container_types>
    using volume_finder =
        grid<axes<cylinder3D, axis::bounds::e_open, axis::irregular,
                  axis::regular, axis::irregular>,
             bins::single<dindex>, simple_serializer, container_t>;
};

}  // namespace detray
