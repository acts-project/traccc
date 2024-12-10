/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/builders/grid_builder.hpp"
#include "detray/builders/homogeneous_material_builder.hpp"
#include "detray/builders/homogeneous_volume_material_builder.hpp"
#include "detray/builders/material_map_builder.hpp"
#include "detray/builders/volume_builder.hpp"
#include "detray/core/detail/container_buffers.hpp"
#include "detray/core/detail/container_views.hpp"
#include "detray/core/detail/surface_lookup.hpp"
#include "detray/core/detector_metadata.hpp"
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/containers.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/geometry/detail/volume_descriptor.hpp"

// Vecmem include(s)
#include <vecmem/memory/memory_resource.hpp>

// System include(s)
#include <map>
#include <sstream>
#include <string>

namespace detray {

namespace detail {
/// Temporary way to manipulate transforms in the transform store
/// @todo Remove as soon as contices can be registered!
template <typename detector_t, typename transform3_t>
void set_transform(detector_t &det, const transform3_t &trf, unsigned int i) {
    std::cout
        << "WARNING: Modifying transforms in the detector will be deprecated! "
           "Please, use a separate geometry context in this case"
        << std::endl;
    det._transforms.at(i) = trf;
}
}  // namespace detail

/// @brief The detector definition.
///
/// This class is a heavily templated container aggregation, that owns all data
/// and sets the interface between geometry, navigator and surface finder
/// structures. Its view type is used to move the data between host and device.
///
/// @tparam metadata helper that defines collection and link types centrally
/// @tparam container_t type collection of the underlying containers
template <typename metadata_t = default_metadata,
          typename container_t = host_container_types>
class detector {

    // Allow the building of the detector containers
    friend class volume_builder<detector<metadata_t, container_t>>;
    template <typename, concepts::grid, typename, typename>
    friend class grid_builder;
    friend class homogeneous_material_builder<
        detector<metadata_t, container_t>>;
    friend class homogeneous_volume_material_builder<
        detector<metadata_t, container_t>>;
    template <typename, std::size_t, typename>
    friend class material_map_builder;
    template <typename>
    friend class volume_accelerator_builder;
    /// @todo Remove
    friend void
    detail::set_transform<detector<metadata_t, container_t>,
                          dtransform3D<typename metadata_t::algebra_type>>(
        detector<metadata_t, container_t> &,
        const dtransform3D<typename metadata_t::algebra_type> &, unsigned int);

    public:
    /// Main definition of geometry types
    using metadata = metadata_t;

    /// Algebra types
    using algebra_type = typename metadata::algebra_type;
    using scalar_type = dscalar<algebra_type>;
    using point2_type = dpoint2D<algebra_type>;
    using point3_type = dpoint3D<algebra_type>;
    using vector3_type = dvector3D<algebra_type>;
    using transform3_type = dtransform3D<algebra_type>;

    /// Raw container types
    template <typename T, std::size_t N>
    using array_type = typename container_t::template array_type<T, N>;
    template <typename T>
    using vector_type = typename container_t::template vector_type<T>;
    template <typename... T>
    using tuple_type = typename container_t::template tuple_type<T...>;
    template <typename T>
    using jagged_vector_type =
        typename container_t::template jagged_vector_type<T>;

    /// In case the detector needs to be printed
    using name_map = std::map<dindex, std::string>;

    /// The surface takes a mask (defines the local coordinates and the surface
    /// extent), its material, a link to an element in the transform container
    /// to define its placement and a source link to the object it represents.
    using surface_type = typename metadata::surface_type;
    using surface_container = vector_type<surface_type>;
    using surface_lookup_container = surface_lookup<surface_type, vector_type>;

    /// Forward the alignable transform container (surface placements) and
    /// the geo context (e.g. for alignment)
    using transform_container =
        typename metadata::template transform_store<vector_type>;
    using geometry_context = typename transform_container::context_type;

    /// Forward mask types that are present in this detector
    using mask_container =
        typename metadata::template mask_store<tuple_type, vector_type>;
    using masks = typename mask_container::value_types;

    /// Forward mask types that are present in this detector
    using material_container =
        typename metadata::template material_store<tuple_type, container_t>;
    using materials = typename material_container::value_types;

    /// Surface Finders: structures that enable neigborhood searches in the
    /// detector geometry during navigation. Can be different in each volume
    using accelerator_container =
        typename metadata::template accelerator_store<tuple_type, container_t>;
    using accel = typename accelerator_container::value_types;

    /// Volume type
    using geo_obj_ids = typename metadata::geo_objects;
    using material_link = typename material_container::single_link;
    using accel_link = typename accelerator_container::single_link;
    using volume_type =
        volume_descriptor<geo_obj_ids, accel_link, material_link>;
    using volume_container = vector_type<volume_type>;

    /// Volume finder definition: Make volume index available from track
    /// position
    using volume_finder =
        typename metadata::template volume_finder<container_t>;

    /// Detector view types
    /// @TODO: Switch to const_view_type always if possible
    using view_type = dmulti_view<dvector_view<volume_type>,
                                  typename surface_lookup_container::view_type,
                                  typename transform_container::view_type,
                                  typename mask_container::view_type,
                                  typename material_container::view_type,
                                  typename accelerator_container::view_type,
                                  typename volume_finder::view_type>;

    static_assert(concepts::device_view<view_type>,
                  "Detector view type ill-formed");

    using const_view_type =
        dmulti_view<dvector_view<const volume_type>,
                    typename surface_lookup_container::const_view_type,
                    typename transform_container::const_view_type,
                    typename mask_container::const_view_type,
                    typename material_container::const_view_type,
                    typename accelerator_container::const_view_type,
                    typename volume_finder::const_view_type>;

    static_assert(concepts::device_view<const_view_type>,
                  "Detector const view type ill-formed");

    /// Detector buffer types
    using buffer_type =
        dmulti_buffer<dvector_buffer<volume_type>,
                      typename surface_lookup_container::buffer_type,
                      typename transform_container::buffer_type,
                      typename mask_container::buffer_type,
                      typename material_container::buffer_type,
                      typename accelerator_container::buffer_type,
                      typename volume_finder::buffer_type>;

    static_assert(concepts::device_buffer<buffer_type>,
                  "Detector buffer type ill-formed");

    detector() = delete;
    // The detector holds a lot of data and should never be copied
    detector(const detector &) = delete;
    detector &operator=(const detector &) = delete;

    /// Allowed constructors
    /// @{
    /// Move constructor
    detector(detector &&) noexcept = default;

    /// Move assignment
    detector &operator=(detector &&) noexcept = default;

    /// Default construction
    /// @param resource memory resource for the allocation of members
    DETRAY_HOST
    explicit detector(vecmem::memory_resource &resource)
        : _volumes(&resource),
          _surfaces(resource),
          _transforms(resource),
          _masks(resource),
          _materials(resource),
          _accelerators(resource),
          _volume_finder(resource) {}

    /// Constructor from detector data view
    template <concepts::device_view detector_view_t>
    DETRAY_HOST_DEVICE explicit detector(detector_view_t &det_data)
        : _volumes(detray::detail::get<0>(det_data.m_view)),
          _surfaces(detray::detail::get<1>(det_data.m_view)),
          _transforms(detray::detail::get<2>(det_data.m_view)),
          _masks(detray::detail::get<3>(det_data.m_view)),
          _materials(detray::detail::get<4>(det_data.m_view)),
          _accelerators(detray::detail::get<5>(det_data.m_view)),
          _volume_finder(detray::detail::get<6>(det_data.m_view)) {}
    /// @}

    /// @returns a string that contains the detector name
    const std::string &name(const name_map &names) const { return names.at(0); }

    /// @return the sub-volumes of the detector - const access
    DETRAY_HOST_DEVICE
    inline auto volumes() const -> const vector_type<volume_type> & {
        return _volumes;
    }

    /// @return the volume by @param volume_index - const access
    DETRAY_HOST_DEVICE
    inline const auto &volume(dindex volume_index) const {
        return _volumes[volume_index];
    }

    /// @return the volume by global cartesian @param position - const access
    DETRAY_HOST_DEVICE
    inline const auto &volume(const point3_type &p) const {
        // The 3D cylindrical volume search grid is concentric
        const transform3_type identity{};
        const auto loc_pos =
            _volume_finder.project(identity, p, identity.translation());

        // Only one entry per bin
        dindex volume_index{_volume_finder.search(loc_pos).value()};
        return _volumes[volume_index];
    }

    /// @returns all portals - const
    /// @note Depending on the detector type, this can also contain other
    /// surfaces
    /// @todo add range filter to skip non-portal surfaces
    DETRAY_HOST_DEVICE
    inline const auto &portals() const {
        // All portals are registered with the brute force search
        return _accelerators.template get<accel::id::e_brute_force>().all();
    }

    /// @return the sub-volumes of the detector - const access
    DETRAY_HOST_DEVICE
    inline auto surfaces() const -> const surface_lookup_container & {
        return _surfaces;
    }

    /// @returns a surface using a query objetc @param q. This can be an index,
    /// a barcode or a source link searcher (see @c surface_lookup class)
    template <typename query_t>
    DETRAY_HOST_DEVICE constexpr decltype(auto) surface(query_t &&q) const {
        return _surfaces.search(std::forward<query_t>(q));
    }

    /// @return detector transform store
    DETRAY_HOST_DEVICE
    inline auto transform_store(const geometry_context & /*ctx*/ = {}) const
        -> const transform_container & {
        return _transforms;
    }

    /// @return all surface/portal masks in the geometry - const access
    DETRAY_HOST_DEVICE
    inline auto mask_store() const -> const mask_container & { return _masks; }

    /// @return all materials in the geometry - const access
    DETRAY_HOST_DEVICE
    inline auto material_store() const -> const material_container & {
        return _materials;
    }

    /// @returns access to the surface finder container
    DETRAY_HOST_DEVICE
    inline auto accelerator_store() const -> const accelerator_container & {
        return _accelerators;
    }

    /// @return the volume grid - const access
    DETRAY_HOST_DEVICE
    inline auto volume_search_grid() const -> const volume_finder & {
        return _volume_finder;
    }

    /// @returns view of a detector
    DETRAY_HOST auto get_data() -> view_type {
        return view_type{
            detray::get_data(_volumes),      detray::get_data(_surfaces),
            detray::get_data(_transforms),   detray::get_data(_masks),
            detray::get_data(_materials),    detray::get_data(_accelerators),
            detray::get_data(_volume_finder)};
    }

    /// @returns const view of a detector
    DETRAY_HOST auto get_data() const -> const_view_type {
        return const_view_type{
            detray::get_data(_volumes),      detray::get_data(_surfaces),
            detray::get_data(_transforms),   detray::get_data(_masks),
            detray::get_data(_materials),    detray::get_data(_accelerators),
            detray::get_data(_volume_finder)};
    }

    /// @param names maps a volume to its string representation.
    /// @returns a string representation of the detector.
    DETRAY_HOST
    auto to_string(const name_map &names) const -> std::string {
        std::stringstream ss;

        ss << "[>] Detector '" << names.at(0) << "' has " << _volumes.size()
           << " volumes." << std::endl;

        for (const auto [i, v] : detray::views::enumerate(_volumes)) {
            ss << "[>>] Volume at index " << i << ": " << std::endl;
            ss << " - name: '" << names.at(v.index() + 1u) << "'" << std::endl;

            ss << "     contains    "
               << v.template n_objects<geo_obj_ids::e_sensitive>()
               << " sensitive surfaces " << std::endl;

            ss << "                 "
               << v.template n_objects<geo_obj_ids::e_portal>() << " portals "
               << std::endl;

            ss << "                 " << _accelerators.n_collections()
               << " surface finders " << std::endl;

            if (v.accel_index() != dindex_invalid) {
                ss << "  sf finder id " << v.accel_type() << "  sf finders idx "
                   << v.accel_index() << std::endl;
            }
        }

        return ss.str();
    }

    /// Add the volume grid - move semantics
    ///
    /// @param v_grid the volume grid to be added
    DETRAY_HOST
    inline auto set_volume_finder(volume_finder &&v_grid) -> void {
        _volume_finder = std::move(v_grid);
    }

    /// Add the volume grid - copy semantics
    ///
    /// @param v_grid the volume grid to be added
    DETRAY_HOST
    inline auto set_volume_finder(const volume_finder &v_grid) -> void {
        _volume_finder = v_grid;
    }

    private:
    /// Contains the detector sub-volumes.
    volume_container _volumes;

    /// Lookup for surfaces from barcodes
    surface_lookup_container _surfaces;

    /// Keeps all of the transform data in contiguous memory
    transform_container _transforms;

    /// Masks of all surfaces in the geometry in contiguous memory
    mask_container _masks;

    /// Materials of all surfaces in the geometry in contiguous memory
    material_container _materials;

    /// All surface finder data structures that are used in the detector volumes
    accelerator_container _accelerators;

    /// Search structure for volumes
    volume_finder _volume_finder;
};

}  // namespace detray
