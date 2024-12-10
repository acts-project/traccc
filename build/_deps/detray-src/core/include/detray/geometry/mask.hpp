/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/geometry/shapes/cuboid3D.hpp"
#include "detray/navigation/intersection/intersection.hpp"

// System include(s)
#include <algorithm>
#include <cassert>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace detray {

/// @brief Mask a region on a surface and link it to a volume.
///
/// The class uses a lightweight 'shape' that defines the local geometry of a
/// surface (local coordinates and how to check boundaries on local point that
/// lies on the surface). A simple example is @c rectangle2D , can be used to
/// mask a rectangle on an underlying plane.
/// This class can then be instantiated as a concrete mask that holds the
/// boundary values, as well as the link to a particular volume, which is
/// needed during geometry navigation.
///
/// @tparam shape_t underlying geometrical shape of the mask, rectangle etc
/// @tparam links_t the type of link into the volume container
///                 (e.g. single index vs range)
template <typename shape_t, typename links_t = std::uint_least16_t,
          typename algebra_t = ALGEBRA_PLUGIN<detray::scalar>>
class mask {
    public:
    // Linear algebra types
    using algebra_type = algebra_t;
    using scalar_type = dscalar<algebra_t>;
    using point2_type = dpoint2D<algebra_t>;
    using point3_type = dpoint3D<algebra_t>;
    using vector3_type = dvector3D<algebra_t>;

    using links_type = links_t;
    using shape = shape_t;
    using boundaries = typename shape::boundaries;
    using mask_values = typename shape::template bounds_type<scalar_type>;
    using local_frame = typename shape::template local_frame_type<algebra_t>;

    /// Default constructor
    constexpr mask() = default;

    /// Constructor from single mask boundary values @param args and
    /// volume link @param link
    template <typename... Args>
    requires(sizeof...(Args) == shape::boundaries::e_size) DETRAY_HOST_DEVICE
        explicit constexpr mask(const links_type& link, Args&&... args)
        : _values({{std::forward<Args>(args)...}}), _volume_link(link) {}

    /// Constructor from mask boundary array @param values and
    /// volume link @param link
    DETRAY_HOST_DEVICE
    constexpr mask(const mask_values& values, const links_type& link)
        : _values{values}, _volume_link{link} {}

    /// Constructor from mask boundary vector @param values and
    /// volume link @param link
    DETRAY_HOST mask(const std::vector<scalar_type>& values,
                     const links_type& link)
        : _volume_link(link) {
        assert(values.size() == boundaries::e_size &&
               " Given number of boundaries does not match mask shape.");
        std::ranges::copy(std::cbegin(values), std::cend(values),
                          std::begin(_values));
    }

    /// Assignment operator from an array, convenience function
    ///
    /// @param rhs is the right hand side object
    DETRAY_HOST
    auto operator=(const mask_values& rhs)
        -> mask<shape_t, links_t, algebra_t>& {
        _values = rhs;
        return (*this);
    }

    /// Equality operator
    ///
    /// @param rhs is the mask to be compared
    ///
    /// @returns a boolean if the values and links are equal.
    bool operator==(const mask& rhs) const = default;

    /// Subscript operator - non-const
    ///
    /// @returns the reference to the member variable
    DETRAY_HOST_DEVICE
    constexpr auto operator[](const std::size_t value_index) -> scalar_type& {
        return _values[value_index];
    }

    /// Subscript operator - const
    ///
    /// @returns a copy of the member variable
    DETRAY_HOST_DEVICE
    constexpr auto operator[](const std::size_t value_index) const
        -> scalar_type {
        return _values[value_index];
    }

    /// @returns the mask shape
    DETRAY_HOST_DEVICE
    static consteval auto get_shape() -> shape { return shape{}; }

    /// @returns return local frame object (used in geometrical checks)
    DETRAY_HOST_DEVICE static consteval local_frame get_local_frame() {
        return local_frame{};
    }

    /// @returns the global point projected onto the surface
    template <typename transform3D_t>
    DETRAY_HOST_DEVICE inline static auto to_local_frame(
        const transform3D_t& trf, const point3_type& glob_p,
        const point3_type& glob_dir = {}) -> point3_type {
        return get_local_frame().global_to_local_3D(trf, glob_p, glob_dir);
    }

    /// @returns the global point for a local position on the surface
    template <typename transform3D_t>
    DETRAY_HOST_DEVICE inline static auto to_global_frame(
        const transform3D_t& trf, const point3_type& loc) -> point3_type {
        return get_local_frame().local_to_global(trf, loc);
    }

    /// @brief Mask this shape onto a surface.
    ///
    /// @note the point is expected to be given in local coordinates and to lie
    /// on the underlying surface geometry.
    /// For the projection from global to local coordinates, the method
    /// @c to_local_frame() can be used.
    /// To check that a point lies on the surface, use the corresponding
    /// intersector.
    ///
    /// @param loc_p the point to be checked in the local system
    /// @param tol dynamic tolerance determined by caller
    ///
    /// @return an intersection status e_inside / e_outside
    DETRAY_HOST_DEVICE
    inline auto is_inside(
        const point3_type& loc_p,
        const scalar_type t =
            std::numeric_limits<scalar_type>::epsilon()) const {

        return get_shape().check_boundaries(_values, loc_p, t);
    }

    /// @returns the boundary values
    DETRAY_HOST_DEVICE
    auto values() const -> const mask_values& { return _values; }

    /// @returns the volume link - const reference
    DETRAY_HOST_DEVICE
    auto volume_link() const -> const links_type& { return _volume_link; }

    /// @returns the volume link - non-const access
    DETRAY_HOST_DEVICE
    auto volume_link() -> links_type& { return _volume_link; }

    /// @returns the masks measure (area or volume covered by boundary check
    /// on local positions)
    DETRAY_HOST_DEVICE constexpr auto measure() const -> scalar_type {
        return get_shape().measure(_values);
    }

    /// @returns the area the mask defines on the local geometry (2D)
    template <typename S = shape_t>
    requires(S::dim == 2) DETRAY_HOST_DEVICE constexpr auto area() const
        -> scalar_type {
        return get_shape().area(_values);
    }

    /// @returns the area the mask defines on the local geometry (3D)
    template <typename S = shape_t>
    requires(S::dim == 3) DETRAY_HOST_DEVICE constexpr auto volume() const
        -> scalar_type {
        return get_shape().volume(_values);
    }

    /// @returns the masks centroid in local cartesian coordinates
    DETRAY_HOST_DEVICE auto centroid() const {
        return get_shape().template centroid<algebra_t>(_values);
    }

    /// @brief Find the minimum distance to any boundary.
    ///
    /// @param loc_p the point to be checked in the local system
    ///
    /// @returns the minimum distance.
    DETRAY_HOST_DEVICE
    auto min_dist_to_boundary(const point3_type& loc_p) const -> scalar_type {
        return get_shape().min_dist_to_boundary(_values, loc_p);
    }

    /// @brief Lower and upper point for minimum axis aligned bounding box.
    ///
    /// Computes the min and max vertices in a local 3 dim cartesian frame.
    ///
    /// @param env dynamic envelope around the shape
    ///
    /// @returns a cuboid3D mask that is equivalent to the minimum local aabb.
    DETRAY_HOST_DEVICE
    auto local_min_bounds(const scalar_type env =
                              std::numeric_limits<scalar_type>::epsilon()) const
        -> mask<cuboid3D, unsigned int> {
        const auto bounds =
            get_shape().template local_min_bounds<algebra_t>(_values, env);
        static_assert(bounds.size() == cuboid3D::e_size,
                      "Shape returns incompatible bounds for bound box");
        return {bounds, std::numeric_limits<unsigned int>::max()};
    }

    /// @brief Vertices of the mask in local cartesian coordinates.
    ///
    /// Computes vertices along the mask boundary.
    ///
    /// @param n_seg the number of segments in for arcs
    ///
    /// @returns a vector of vertices.
    DETRAY_HOST
    auto vertices(const dindex n_seg) const -> dvector<point3_type> {
        return get_shape().template vertices<algebra_t>(_values, n_seg);
    }

    /// @returns true if the mask boundary values are consistent
    DETRAY_HOST
    constexpr bool self_check(std::ostream& os) const {

        const bool result = get_shape().check_consistency(_values, os);

        if (!result) {
            os << to_string();
        }

        return result;
    }

    /// @returns a string representation of the mask
    DETRAY_HOST
    auto to_string() const -> std::string {
        std::stringstream ss;
        ss << std::string(shape::name);
        for (const auto& v : _values) {
            ss << ", " << v;
        }
        return ss.str();
    }

    /// @returns a string stream that prints the mask details
    DETRAY_HOST
    friend std::ostream& operator<<(std::ostream& os, const mask& m) {
        os << m.to_string();
        return os;
    }

    private:
    mask_values _values{};
    links_type _volume_link{std::numeric_limits<links_type>::max()};
};

}  // namespace detray
