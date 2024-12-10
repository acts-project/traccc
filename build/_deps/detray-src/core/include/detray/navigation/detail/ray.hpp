/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/tracks/free_track_parameters.hpp"

// System include(s).
#include <ostream>

namespace detray::detail {

/// @brief describes a straight-line trajectory
template <typename algebra_t>
class ray {
    public:
    using algebra_type = algebra_t;
    using scalar_type = dscalar<algebra_type>;
    using point3_type = dpoint3D<algebra_type>;
    using vector3_type = dvector3D<algebra_type>;
    using transform3_type = dtransform3D<algebra_type>;

    using free_track_parameters_type = free_track_parameters<algebra_t>;
    using free_vector_type = typename free_track_parameters_type::vector_type;

    ray() = default;

    /// Parametrized constructor from a position and direction
    ///
    /// @param pos the track position
    /// @param dir the track momentum direction
    DETRAY_HOST_DEVICE ray(point3_type &&pos, vector3_type &&dir)
        : _pos{std::move(pos)}, _dir{std::move(dir)} {}

    /// Parametrized constructor that complies with track interface
    ///
    /// @param pos the track position
    /// @param dir the track momentum direction
    DETRAY_HOST_DEVICE ray(const point3_type &pos, const scalar_type /*time*/,
                           const vector3_type &dir, const scalar_type /*qop*/)
        : _pos{pos}, _dir{dir} {}

    /// Parametrized constructor that complies with track interface
    ///
    /// @param track the track state that should be approximated
    DETRAY_HOST_DEVICE explicit ray(const free_track_parameters_type &track)
        : ray(track.pos(), track.dir()) {}

    /// @returns position on the ray (compatible with tracks/intersectors)
    DETRAY_HOST_DEVICE point3_type pos() const { return _pos; }

    /// @returns position on the ray paramterized by path length
    DETRAY_HOST_DEVICE point3_type pos(const scalar_type s) const {
        // Direction is always normalized in the constructor
        return _pos + s * _dir;
    }

    /// @param position new position on the ray
    DETRAY_HOST_DEVICE void set_pos(point3_type pos) { _pos = pos; }

    /// @returns direction of the ray (compatible with tracks/intersectors)
    DETRAY_HOST_DEVICE vector3_type dir() const { return _dir; }

    /// @returns direction of the ray paramterized by path length
    DETRAY_HOST_DEVICE vector3_type dir(const scalar_type /*s*/) const {
        return this->dir();
    }

    /// @param dir new direction of the ray
    DETRAY_HOST_DEVICE void set_dir(vector3_type dir) { _dir = dir; }

    /// @returns the q over p value: Zero for ray
    DETRAY_HOST_DEVICE
    constexpr scalar_type qop() const { return 0.f; }

    /// Print
    DETRAY_HOST
    friend std::ostream &operator<<(std::ostream &os, const ray &r) {
        os << "ray: ";
        os << "ori = [" << r._pos[0] << ", " << r._pos[1] << ", " << r._pos[2]
           << "], ";
        os << "dir = [" << r._dir[0] << ", " << r._dir[1] << ", " << r._dir[2]
           << "]" << std::endl;

        return os;
    }

    private:
    /// origin of ray
    point3_type _pos{0.f, 0.f, 0.f};
    /// direction of ray
    vector3_type _dir{0.f, 0.f, 1.f};
};

}  // namespace detray::detail
