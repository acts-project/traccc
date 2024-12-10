/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/units.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes.hpp"
#include "detray/geometry/tracking_surface.hpp"
#include "detray/geometry/tracking_volume.hpp"

// System include(s)
#include <cassert>
#include <optional>

namespace detray::svgtools::utils {

/// @brief Functor to calculate the outermost radius a shape.
/// If the shape is not defined by a radius, then null option is returned.
struct outer_radius_getter {

    public:
    template <typename mask_group_t, typename index_t>
    DETRAY_HOST inline std::optional<detray::scalar> operator()(
        const mask_group_t& mask_group, const index_t& index) const {
        return outer_radius(mask_group[index]);
    }

    private:
    // The remaining shapes do not have an outer radius.
    template <typename mask_t>
    std::optional<detray::scalar> inline outer_radius(
        const mask_t& /*mask*/) const {
        return std::nullopt;
    }

    // Calculates the outer radius for rings.
    auto inline outer_radius(const detray::mask<detray::ring2D>& mask) const {
        return std::optional<detray::scalar>(mask[ring2D::e_outer_r]);
    }

    // Calculates the outer radius for annuluses.
    auto inline outer_radius(
        const detray::mask<detray::annulus2D>& mask) const {
        return std::optional<detray::scalar>(mask[annulus2D::e_max_r]);
    }

    // Calculates the outer radius for cylinders (2D).
    auto inline outer_radius(
        const detray::mask<detray::cylinder2D>& mask) const {
        return std::optional<detray::scalar>(mask[cylinder2D::e_r]);
    }

    // Calculates the outer radius for concentric cylinders (2D).
    auto inline outer_radius(
        const detray::mask<detray::concentric_cylinder2D>& mask) const {
        return std::optional<detray::scalar>(mask[concentric_cylinder2D::e_r]);
    }

    // Calculates the outer radius for cylinders (3D).
    auto inline outer_radius(
        const detray::mask<detray::cylinder3D>& mask) const {
        return std::optional<detray::scalar>(mask[cylinder3D::e_max_r]);
    }
};

/// @brief Functor to obtain the volume link.
struct link_getter {
    template <typename mask_group_t, typename index_t>
    DETRAY_HOST inline auto operator()(const mask_group_t& mask_group,
                                       const index_t& index) const {
        return mask_group[index].volume_link();
    }
};

/// @brief Functor to calculate a suitable starting point for displaying the
/// link arrow.
struct link_start_getter {

    public:
    template <typename mask_group_t, typename index_t, typename transform_t>
    DETRAY_HOST inline auto operator()(const mask_group_t& mask_group,
                                       const index_t& index,
                                       const transform_t& transform) const {
        return link_start(mask_group[index], transform);
    }

    private:
    // Calculates the link starting location of the remaining shapes.
    template <typename mask_t, typename transform_t>
    auto inline link_start(const mask_t& mask,
                           const transform_t& transform) const {
        return transform.point_to_global(mask.centroid());
    }

    // Calculates the (optimal) link starting point for rings.
    template <typename transform_t>
    auto inline link_start(const detray::mask<detray::ring2D>& mask,
                           const transform_t& transform) const {

        using shape_t = detray::ring2D;
        using mask_t = detray::mask<shape_t>;
        using point3_t = typename mask_t::point3_type;
        using scalar_t = typename mask_t::scalar_type;

        const scalar_t r{0.5f *
                         (mask[shape_t::e_inner_r] + mask[shape_t::e_outer_r])};
        const scalar_t phi{detray::constant<scalar_t>::pi_2};

        return mask_t::to_global_frame(transform, point3_t{r, phi, 0.f});
    }

    // Calculates the (optimal) link starting point for annuluses.
    template <typename transform_t>
    auto inline link_start(const detray::mask<detray::annulus2D>& mask,
                           const transform_t& transform) const {

        using shape_t = detray::annulus2D;
        using mask_t = detray::mask<shape_t>;
        using point3_t = typename mask_t::point3_type;
        using scalar_t = typename mask_t::scalar_type;

        const scalar_t r{(mask[shape_t::e_min_r] + mask[shape_t::e_max_r]) /
                         2.f};
        const scalar_t phi{mask[shape_t::e_average_phi]};

        return mask_t::to_global_frame(transform, point3_t{r, phi, 0.f});
    }

    // Calculates the (optimal) link starting point for concentric cylinders
    template <typename transform_t>
    auto inline link_start(const detray::mask<concentric_cylinder2D>& mask,
                           const transform_t& transform) const {
        using mask_t = detray::mask<concentric_cylinder2D>;
        using point3_t = typename mask_t::point3_type;
        using scalar_t = typename mask_t::scalar_type;

        const scalar_t r{mask[concentric_cylinder2D::e_r]};
        const scalar_t phi{detray::constant<scalar_t>::pi_2};
        // Shift the center to the actual cylinder bounds
        const scalar_t z{mask.centroid()[2]};

        return mask_t::to_global_frame(transform, point3_t{phi, z, r});
    }

    // Calculates the (optimal) link starting point for cylinders (2D).
    template <typename transform_t>
    auto inline link_start(const detray::mask<cylinder2D>& mask,
                           const transform_t& transform) const {
        using mask_t = detray::mask<cylinder2D>;
        using point3_t = typename mask_t::point3_type;
        using scalar_t = typename mask_t::scalar_type;

        const scalar_t r{mask[cylinder2D::e_r]};
        const scalar_t phi{detray::constant<scalar_t>::pi_2};
        // Shift the center to the actual cylinder bounds
        const scalar_t z{mask.centroid()[2]};

        return mask_t::to_global_frame(transform, point3_t{r * phi, z, r});
    }

    // Calculates the (optimal) link starting point for cylinders (3D).
    template <typename transform_t>
    auto inline link_start(const detray::mask<detray::cylinder3D>& mask,
                           const transform_t& transform) const {

        using shape_t = detray::cylinder3D;
        using mask_t = detray::mask<shape_t>;
        using point3_t = typename mask_t::point3_type;
        using scalar_t = typename mask_t::scalar_type;

        const scalar_t r{mask[shape_t::e_max_r]};
        const scalar_t phi{
            0.5f * (mask[shape_t::e_max_phi] + mask[shape_t::e_max_phi])};
        const scalar_t z{mask.centroid()[2]};

        return mask_t::to_global_frame(transform, point3_t{r, phi, z});
    }
};

/// @brief Functor to calculate a suitable end point for displaying the link
/// arrow.
struct link_end_getter {

    public:
    template <typename mask_group_t, typename index_t, typename detector_t,
              typename point3_t, typename vector3_t, typename scalar_t>
    DETRAY_HOST inline auto operator()(
        const mask_group_t& mask_group, const index_t& index,
        const detector_t& detector,
        const detray::tracking_volume<detector_t>& volume,
        const point3_t& surface_point, const vector3_t& surface_normal,
        const scalar_t& link_length) const {

        return link_dir(mask_group[index], detector, volume, surface_point,
                        surface_normal) *
                   link_length +
               surface_point;
    }

    private:
    /// @brief Calculates the direction of the link for remaining shapes.
    template <typename detector_t, typename mask_t, typename point3_t,
              typename vector3_t>
    inline auto link_dir(const mask_t& /*mask*/, const detector_t& /*detector*/,
                         const detray::tracking_volume<detector_t>& volume,
                         const point3_t& surface_point,
                         const vector3_t& surface_normal) const {
        const auto dir = volume.center() - surface_point;
        const auto dot_prod = vector::dot(dir, surface_normal);

        // Should geometrically not happen with a local point 'surface_point'
        assert(dot_prod != 0.f);

        return math::copysign(1.f, dot_prod) * surface_normal;
    }

    /// @brief Calculates the direction of the link for cylinders (2D)
    template <typename detector_t, typename point3_t, typename vector3_t,
              typename shape_t>
        requires std::is_same_v<shape_t, cylinder2D> ||
        std::is_same_v<shape_t, concentric_cylinder2D> inline auto link_dir(
            const detray::mask<shape_t>& mask, const detector_t& detector,
            const detray::tracking_volume<detector_t>& volume,
            const point3_t& /*surface_point*/,
            const vector3_t& surface_normal) const {
        for (const auto& desc : volume.portals()) {

            const detray::tracking_surface surface{detector, desc};

            if (auto r = surface.template visit_mask<outer_radius_getter>()) {
                if (*r > mask[shape_t::e_r]) {
                    return surface_normal;
                }
            }
        }
        return -1.f * surface_normal;
    }
};

}  // namespace detray::svgtools::utils
