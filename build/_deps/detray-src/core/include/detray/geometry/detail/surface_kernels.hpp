/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/materials/detail/concepts.hpp"
#include "detray/materials/detail/material_accessor.hpp"
#include "detray/materials/material.hpp"
#include "detray/propagator/detail/jacobian_engine.hpp"
#include "detray/tracks/detail/transform_track_parameters.hpp"
#include "detray/tracks/tracks.hpp"

// System include(s)
#include <limits>
#include <ostream>

namespace detray::detail {

/// Functors to be used in the @c surface class
template <typename algebra_t>
struct surface_kernels {

    using algebra_type = algebra_t;
    using scalar_type = dscalar<algebra_t>;
    using point2_type = dpoint2D<algebra_t>;
    using point3_type = dpoint3D<algebra_t>;
    using vector3_type = dvector3D<algebra_t>;
    using transform3_type = dtransform3D<algebra_t>;
    using bound_param_vector_type = bound_parameters_vector<algebra_t>;
    using free_param_vector_type = free_parameters_vector<algebra_t>;
    using free_matrix_type = free_matrix<algebra_t>;

    /// A functor to retrieve the masks shape name
    struct get_shape_name {
        template <typename mask_group_t, typename index_t>
        DETRAY_HOST inline std::string operator()(const mask_group_t&,
                                                  const index_t&) const {

            return std::string(mask_group_t::value_type::shape::name);
        }
    };

    /// A functor that checks if a local point @param loc_p is within the
    /// surface mask with tolerance @param tol
    struct is_inside {
        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline bool operator()(
            const mask_group_t& mask_group, const index_t& index,
            const point3_type& loc_p, const scalar_type tol) const {

            return mask_group[index].is_inside(loc_p, tol);
        }
    };

    /// A functor to run the mask self check. Puts error messages into @param os
    struct mask_self_check {
        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline auto operator()(
            const mask_group_t& mask_group, const index_t& index,
            std::ostream& os) const {

            return mask_group.at(index).self_check(os);
        }
    };

    /// A functor to retrieve the masks volume link
    struct get_volume_link {
        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline auto operator()(
            const mask_group_t& mask_group, const index_t& index) const {

            return mask_group[index].volume_link();
        }
    };

    /// A functor to retrieve a mask boundary, determined by @param i
    struct get_mask_value {
        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline auto operator()(
            const mask_group_t& mask_group, const index_t& index,
            std::size_t i) const {

            return mask_group[index][i];
        }
    };

    /// A functor to retrieve the mask boundaries (host only)
    struct get_mask_values {
        template <typename mask_group_t, typename index_t>
        DETRAY_HOST inline auto operator()(const mask_group_t& mask_group,
                                           const index_t& index) const {

            std::vector<scalar_type> values{};
            for (const scalar_type v : mask_group[index].values()) {
                values.push_back(v);
            }

            return values;
        }
    };

    /// A functor to retrieve the material parameters
    struct get_material_params {
        template <typename mat_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline auto operator()(
            const mat_group_t& mat_group, const index_t& idx,
            const point2_type& loc_p) const {

            using material_t = typename mat_group_t::value_type;

            if constexpr (concepts::surface_material<material_t>) {
                return &(detail::material_accessor::get(mat_group, idx, loc_p)
                             .get_material());
            } else {
                using scalar_t = typename material_t::scalar_type;
                // Volume material (cannot be reached from a surface)
                return static_cast<const material<scalar_t>*>(nullptr);
            }
        }
    };

    /// A functor get the surface normal at a given local/bound position
    struct normal {
        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline point3_type operator()(
            const mask_group_t& mask_group, const index_t& index,
            const transform3_type& trf3, const point2_type& bound) const {
            using mask_t = typename mask_group_t::value_type;

            return mask_t::get_local_frame().normal(trf3, bound,
                                                    mask_group[index]);
        }

        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline point3_type operator()(
            const mask_group_t& /*mask_group*/, const index_t& /*index*/,
            const transform3_type& trf3, const point3_type& local) const {
            using mask_t = typename mask_group_t::value_type;

            return mask_t::get_local_frame().normal(trf3, local);
        }
    };

    /// A functor get the mask centroid in local cartesian coordinates
    struct centroid {
        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline point3_type operator()(
            const mask_group_t& mask_group, const index_t& index) const {

            return mask_group[index].centroid();
        }
    };

    /// A functor to perform global to local bound transformation
    struct global_to_bound {
        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline point2_type operator()(
            const mask_group_t& /*mask_group*/, const index_t& /*index*/,
            const transform3_type& trf3, const point3_type& global,
            const vector3_type& dir) const {
            using mask_t = typename mask_group_t::value_type;

            const point3_type local = mask_t::to_local_frame(trf3, global, dir);

            return {local[0], local[1]};
        }
    };

    /// A functor to perform global to local transformation
    struct global_to_local {
        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline point3_type operator()(
            const mask_group_t& /*mask_group*/, const index_t& /*index*/,
            const transform3_type& trf3, const point3_type& global,
            const vector3_type& dir) const {
            using mask_t = typename mask_group_t::value_type;

            return mask_t::to_local_frame(trf3, global, dir);
        }
    };

    /// A functor to perform local to global transformation
    struct local_to_global {

        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline point3_type operator()(
            const mask_group_t& mask_group, const index_t& index,
            const transform3_type& trf3, const point2_type& bound,
            const vector3_type& dir) const {
            using mask_t = typename mask_group_t::value_type;

            return mask_t::get_local_frame().local_to_global(
                trf3, mask_group[index], bound, dir);
        }

        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline point3_type operator()(
            const mask_group_t& /*mask_group*/, const index_t& /*index*/,
            const transform3_type& trf3, const point3_type& local,
            const vector3_type&) const {
            using mask_t = typename mask_group_t::value_type;

            return mask_t::to_global_frame(trf3, local);
        }
    };

    /// A functor to get from a free to a bound vector
    struct free_to_bound_vector {

        // Visitor to the detector mask store that is called on the mask
        // collection that contains the mask (shape) type of the surface
        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline bound_param_vector_type operator()(
            const mask_group_t& /*mask_group*/, const index_t& /*index*/,
            const transform3_type& trf3,
            const free_param_vector_type& free_vec) const {

            using frame_t = typename mask_group_t::value_type::local_frame;

            return detail::free_to_bound_vector<frame_t>(trf3, free_vec);
        }
    };

    /// A functor to get from a bound to a free vector
    struct bound_to_free_vector {

        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline free_param_vector_type operator()(
            const mask_group_t& mask_group, const index_t& index,
            const transform3_type& trf3,
            const bound_param_vector_type& bound_vec) const {

            return detail::bound_to_free_vector(trf3, mask_group[index],
                                                bound_vec);
        }
    };

    /// A functor to get the free-to-bound Jacobian
    struct free_to_bound_jacobian {

        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline auto operator()(
            const mask_group_t& /*mask_group*/, const index_t& /*index*/,
            const transform3_type& trf3,
            const free_param_vector_type& free_vec) const {

            using frame_t = typename mask_group_t::value_type::local_frame;

            return detail::jacobian_engine<frame_t>::free_to_bound_jacobian(
                trf3, free_vec);
        }
    };

    /// A functor to get the bound-to-free Jacobian
    struct bound_to_free_jacobian {

        template <typename mask_group_t, typename index_t>
        DETRAY_HOST_DEVICE inline auto operator()(
            const mask_group_t& mask_group, const index_t& index,
            const transform3_type& trf3,
            const bound_param_vector_type& bound_vec) const {

            using frame_t = typename mask_group_t::value_type::local_frame;

            return detail::jacobian_engine<frame_t>::bound_to_free_jacobian(
                trf3, mask_group[index], bound_vec);
        }
    };

    /// A functor to get the path correction
    struct path_correction {

        template <typename mask_group_t, typename index_t, typename scalar_t>
        DETRAY_HOST_DEVICE inline free_matrix_type operator()(
            const mask_group_t& /*mask_group*/, const index_t& /*index*/,
            const transform3_type& trf3, const vector3_type& pos,
            const vector3_type& dir, const vector3_type& dtds,
            const scalar_t dqopds) const {

            using frame_t = typename mask_group_t::value_type::local_frame;

            return detail::jacobian_engine<frame_t>::path_correction(
                pos, dir, dtds, dqopds, trf3);
        }
    };

    /// A functor to get the local min bounds.
    struct local_min_bounds {

        template <typename mask_group_t, typename index_t, typename scalar_t>
        DETRAY_HOST_DEVICE inline auto operator()(
            const mask_group_t& mask_group, const index_t& index,
            const scalar_t env =
                std::numeric_limits<scalar_t>::epsilon()) const {

            return mask_group[index].local_min_bounds(env);
        }
    };

    /// A functor to get the minimum distance to any surface boundary.
    struct min_dist_to_boundary {

        template <typename mask_group_t, typename index_t, typename point_t>
        DETRAY_HOST_DEVICE inline auto operator()(
            const mask_group_t& mask_group, const index_t& index,
            const point_t& loc_p) const {

            return mask_group[index].min_dist_to_boundary(loc_p);
        }
    };

    /// A functor to get the vertices in local coordinates.
    struct local_vertices {

        template <typename mask_group_t, typename index_t, typename scalar_t>
        DETRAY_HOST_DEVICE inline auto operator()(
            const mask_group_t& mask_group, const index_t& index,
            const dindex n_seg) const {

            return mask_group[index].vertices(n_seg);
        }
    };
};

}  // namespace detray::detail
