/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
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
#include "detray/io/common/detail/type_info.hpp"  // mask_info
#include "detray/materials/material_slab.hpp"
#include "detray/navigation/accelerators/brute_force_finder.hpp"

// Linear algebra types
#include "detray/tutorial/types.hpp"

// New geometric shape type
#include "my_square2D.hpp"

/// This example defines a detray geometry type for a detector with cuboid
/// volumes that constain a new surface shape (squares), trapezoids and
/// rectangles for the volume boundary surfaces (portals).
/// For now, all surfaces in the volume(s) are stored in a 'brute force'
/// acceleration data structure which tests all surfaces it contains during
/// navigation.
/// In this example detector design, volumes do not contain other volumes, so
/// the volume lookup is done using a uniform grid.
/// Furthermore, the detector will contain homogeneous material on its surfaces.
namespace detray {

namespace tutorial {

//
// Surface Primitives
//

/// Portal link type between volumes
using nav_link = std::uint_least16_t;

/// The mask types for the detector sensitive/passive surfaces
using square = mask<square2D, nav_link>;
using trapezoid = mask<trapezoid2D, nav_link>;
// Types for portals
using rectangle = mask<rectangle2D, nav_link>;

//
// Material Description
//

/// The material types to be mapped onto the surfaces: Here homogeneous material
using slab = material_slab<detray::scalar>;

//
// Detector
//

/// Defines a detector that contains squares, trapezoids and a bounding portal
/// box.
struct my_metadata {

    /// Define the algebra type for the geometry and navigation
    using algebra_type = detray::tutorial::algebra_t;

    /// How to index the constituent objects in a volume
    /// If they share the same index value here, they will be added into the
    /// same acceleration data structure in every respective volume
    enum geo_objects : std::uint_least8_t {
        e_surface = 0u,  //< This detector keeps all surfaces in the same
                         //  acceleration data structure (id 0)
        e_size = 1u
    };

    /// How a volume finds its constituent objects in the detector containers
    /// In this case: One range for sensitive/passive surfaces, oportals
    using object_link_type = dmulti_index<dindex_range, geo_objects::e_size>;

    /// How to store and link transforms. The geometry context allows to resolve
    /// the conditions data for e.g. module alignment
    template <template <typename...> class vector_t = dvector>
    using transform_store =
        single_store<transform3, vector_t, geometry_context>;

    /// Assign the mask types to the mask tuple container entries. It may be a
    /// good idea to have the most common types in the first tuple entries, in
    /// order to minimize the depth of the 'unrolling' before a mask is found
    /// in the tuple
    enum class mask_ids : std::uint_least8_t {
        e_square2 = 0,
        e_trapezoid2 = 1,
        e_portal_rectangle2 = 2
    };

    /// This is the mask collections tuple (in the detector called 'mask store')
    /// the @c regular_multi_store is a vecemem-ready tuple of vectors of
    /// the detector masks.
    template <template <typename...> class tuple_t = dtuple,
              template <typename...> class vector_t = dvector>
    using mask_store =
        regular_multi_store<mask_ids, empty_context, tuple_t, vector_t, square,
                            trapezoid, rectangle>;

    /// Similar to the mask store, there is a material store, which
    enum class material_ids : std::uint_least8_t {
        e_slab = 0,
        e_none = 1,
    };

    /// How to store and link materials. The material does not make use of
    /// conditions data ( @c empty_context )
    template <template <typename...> class tuple_t = dtuple,
              typename container_t = host_container_types>
    using material_store =
        multi_store<material_ids, empty_context, tuple_t,
                    typename container_t::template vector_type<slab>>;

    /// Surface descriptor type used for sensitives, passives and portals
    /// It holds the indices to the surface data in the detector data stores
    /// that were defined above
    using transform_link = typename transform_store<>::link_type;
    using mask_link = typename mask_store<>::single_link;
    using material_link = typename material_store<>::single_link;
    using surface_type =
        surface_descriptor<mask_link, material_link, transform_link, nav_link>;

    /// The acceleration data structures live in another tuple that needs to
    /// indexed correctly
    enum class accel_ids : std::uint_least8_t {
        e_brute_force = 0,  //< test all surfaces in a volume (brute force)
        e_default = e_brute_force,
    };

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

}  // namespace tutorial

namespace detail {

/// If the new square shape should participate in the file IO, then detray
/// needs a specialization of the @c mask_info trait, in order to be
/// able to match the gloabl IO id for the new square shape to the static
/// detector mask store that is defined in the metadata above.
/// Of course, the IO id for the square has to be added to the global
/// @c mask_shape enum, too. These mask_shape IDs are global to all detectors
/// and shared with ACTS.
///
/// Please change the following lines in
/// 'detray/io/common/detail/definitions.hpp':
///
/// enum class mask_shape : unsigned int {
///    annulus2 = 0u,
///    ...
///    square2 = 9u,  //< new shape
///    n_shapes = 10u //< The total number of known shapes needs to be raised
///  };
///
/// In order to write the square shape to file:
/// 'detray/io/common/geometery_writer.hpp'
/// ...
/// } else if (name == "square2D") {
///     mask_data.shape = shape_id::square2;
/// } else {
///

/// During the IO, check for a 2D square shape
/*template <typename detector_t>
struct mask_info<io::shape_id::square2, detector_t>
    requires detector_t::masks::template is_defined<
                                      detray::tutorial::square>()> {
    using type = detray::tutorial::square::shape;
    // This mask id is defined in the metadat down below and determines the
    // position of the collection of square in the detector mask tuple (store)
    static constexpr
        typename detector_t::masks::id value{detector_t::masks::id::e_square2};
};*/

}  // namespace detail

}  // namespace detray
