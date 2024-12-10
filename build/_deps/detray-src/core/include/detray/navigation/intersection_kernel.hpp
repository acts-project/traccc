/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/algorithms.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/units.hpp"
#include "detray/navigation/intersection/intersection.hpp"
#include "detray/utils/ranges.hpp"

namespace detray {

/// A functor to add all valid intersections between the trajectory and surface
template <template <typename, typename, bool> class intersector_t>
struct intersection_initialize {

    /// Operator function to initalize intersections
    ///
    /// @tparam mask_group_t is the input mask group type found by variadic
    /// unrolling
    /// @tparam is_container_t is the intersection container type
    /// @tparam traj_t is the input trajectory type (e.g. ray or helix)
    /// @tparam surface_t is the input surface type
    /// @tparam transform_container_t is the input transform store type
    ///
    /// @param mask_group is the input mask group
    /// @param is_container is the intersection container to be filled
    /// @param traj is the input trajectory
    /// @param surface is the input surface
    /// @param contextual_transforms is the input transform container
    /// @param mask_tolerance is the tolerance for mask size
    /// @param overstep_tol negative cutoff for the path
    ///
    /// @return the number of valid intersections
    template <typename mask_group_t, typename mask_range_t,
              typename is_container_t, typename traj_t, typename surface_t,
              typename transform_container_t, typename scalar_t>
    DETRAY_HOST_DEVICE inline void operator()(
        const mask_group_t &mask_group, const mask_range_t &mask_range,
        is_container_t &is_container, const traj_t &traj,
        const surface_t &surface,
        const transform_container_t &contextual_transforms,
        const typename transform_container_t::context_type &ctx,
        const std::array<scalar_t, 2u> &mask_tolerance =
            {0.f, 1.f * unit<scalar_t>::mm},
        const scalar_t mask_tol_scalor = 0.f,
        const scalar_t overstep_tol = 0.f) const {

        using mask_t = typename mask_group_t::value_type;
        using algebra_t = typename mask_t::algebra_type;
        using intersection_t = typename is_container_t::value_type;

        const auto &ctf = contextual_transforms.at(surface.transform(), ctx);

        // Run over the masks that belong to the surface (only one can be hit)
        for (const auto &mask :
             detray::ranges::subrange(mask_group, mask_range)) {

            if (place_in_collection(
                    intersector_t<typename mask_t::shape, algebra_t,
                                  intersection_t::is_debug()>{}(
                        traj, surface, mask, ctf, mask_tolerance,
                        mask_tol_scalor, overstep_tol),
                    is_container)) {
                return;
            }
        }
    }

    private:
    template <typename is_container_t>
    DETRAY_HOST_DEVICE bool place_in_collection(
        const typename is_container_t::value_type &sfi,
        is_container_t &intersections) const {
        if (sfi.status) {
            insert_sorted(sfi, intersections);
        }
        return sfi.status;
    }

    template <typename is_container_t>
    DETRAY_HOST_DEVICE bool place_in_collection(
        std::array<typename is_container_t::value_type, 2> &&solutions,
        is_container_t &intersections) const {
        bool is_valid = false;
        for (auto &sfi : std::move(solutions)) {
            if (sfi.status) {
                insert_sorted(sfi, intersections);
            }
            is_valid |= sfi.status;
        }
        return is_valid;
    }

    template <typename is_container_t>
    DETRAY_HOST_DEVICE void insert_sorted(
        const typename is_container_t::value_type &sfi,
        is_container_t &intersections) const {
        auto itr_pos = detray::detail::upper_bound(intersections.begin(),
                                                   intersections.end(), sfi);
        intersections.insert(itr_pos, sfi);
    }
};

/// A functor to update the closest intersection between the trajectory and
/// surface
template <template <typename, typename, bool> class intersector_t>
struct intersection_update {

    /// Operator function to update the intersection
    ///
    /// @tparam mask_group_t is the input mask group type found by variadic
    /// unrolling
    /// @tparam traj_t is the input trajectory type (e.g. ray or helix)
    /// @tparam surface_t is the input surface type
    /// @tparam transform_container_t is the input transform store type
    ///
    /// @param mask_group is the input mask group
    /// @param mask_range is the range of masks in the group that belong to the
    ///                   surface
    /// @param traj is the input trajectory
    /// @param surface is the input surface
    /// @param contextual_transforms is the input transform container
    /// @param mask_tolerance is the tolerance for mask size
    /// @param overstep_tol negative cutoff for the path
    ///
    /// @return the intersection
    template <typename mask_group_t, typename mask_range_t, typename traj_t,
              typename intersection_t, typename transform_container_t,
              typename scalar_t>
    DETRAY_HOST_DEVICE inline bool operator()(
        const mask_group_t &mask_group, const mask_range_t &mask_range,
        const traj_t &traj, intersection_t &sfi,
        const transform_container_t &contextual_transforms,
        const typename transform_container_t::context_type &ctx,
        const std::array<scalar_t, 2u> &mask_tolerance =
            {0.f, 1.f * unit<scalar_t>::mm},
        const scalar_t mask_tol_scalor = 0.f,
        const scalar_t overstep_tol = 0.f) const {

        using mask_t = typename mask_group_t::value_type;
        using algebra_t = typename mask_t::algebra_type;

        const auto &ctf =
            contextual_transforms.at(sfi.sf_desc.transform(), ctx);

        // Run over the masks that belong to the surface
        for (const auto &mask :
             detray::ranges::subrange(mask_group, mask_range)) {

            intersector_t<typename mask_t::shape, algebra_t,
                          intersection_t::is_debug()>{}
                .update(traj, sfi, mask, ctf, mask_tolerance, mask_tol_scalor,
                        overstep_tol);

            if (sfi.status) {
                return true;
            }
        }

        return false;
    }
};

}  // namespace detray
