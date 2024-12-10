/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/builders/bin_fillers.hpp"
#include "detray/builders/grid_factory.hpp"
#include "detray/builders/surface_factory_interface.hpp"
#include "detray/builders/volume_builder.hpp"
#include "detray/builders/volume_builder_interface.hpp"
#include "detray/geometry/tracking_surface.hpp"
#include "detray/geometry/tracking_volume.hpp"
#include "detray/utils/grid/detail/concepts.hpp"

// System include(s)
#include <array>
#include <cassert>
#include <memory>
#include <vector>

namespace detray {

/// @brief Build a grid of a certain shape.
///
/// Decorator class to a volume builder that adds a grid as the volumes
/// geometry accelerator structure.
template <typename detector_t, concepts::grid grid_t,
          typename bin_filler_t = fill_by_pos,
          typename grid_factory_t = grid_factory_type<grid_t>>
class grid_builder : public volume_decorator<detector_t> {

    using link_id_t = typename detector_t::volume_type::object_id;

    public:
    using scalar_type = typename detector_t::scalar_type;
    using detector_type = detector_t;
    using value_type = typename detector_type::surface_type;

    /// Decorate a volume with a grid
    DETRAY_HOST
    explicit grid_builder(
        std::unique_ptr<volume_builder_interface<detector_t>> vol_builder)
        : volume_decorator<detector_t>(std::move(vol_builder)) {
        // The grid builder provides an acceleration structure to the
        // volume, so don't add sensitive surfaces to the brute force method
        if (this->get_builder()) {
            this->has_accel(true);
        }
    }

    /// Should the passive surfaces be added to the grid ?
    void set_add_passives(bool is_add_passive = true) {
        m_add_passives = is_add_passive;
    }

    /// Set the surface category this grid should contain (type id in the
    /// accelrator link in the volume)
    void set_type(std::size_t sf_id) {
        set_type(static_cast<link_id_t>(sf_id));
    }

    /// Set the surface category this grid should contain (type id in the
    /// accelrator link in the volume)
    void set_type(link_id_t sf_id) {
        // Exclude zero, it is reserved for the brute force method
        assert(static_cast<int>(sf_id) > 0);
        // Make sure the id fits in the volume accelerator link
        assert(sf_id < link_id_t::e_size);

        m_id = sf_id;
    }

    /// Delegate init call depending on @param span type
    template <typename grid_shape_t>
    DETRAY_HOST void init_grid(
        const mask<grid_shape_t> &m,
        const std::array<std::size_t, grid_t::dim> &n_bins,
        const std::vector<std::pair<typename grid_t::loc_bin_index, dindex>>
            &bin_capacities = {},
        const std::array<std::vector<scalar_type>, grid_t::dim> &ax_bin_edges =
            std::array<std::vector<scalar_type>, grid_t::dim>()) {

        static_assert(
            std::is_same_v<typename grid_shape_t::template local_frame_type<
                               typename detector_t::algebra_type>,
                           typename grid_t::local_frame_type>,
            "Mask has incorrect shape");

        m_grid = m_factory.template new_grid<grid_t>(m, n_bins, bin_capacities,
                                                     ax_bin_edges);
    }

    /// Build the empty grid from axis parameters
    DETRAY_HOST void init_grid(
        const std::vector<scalar_type> &spans,
        const std::vector<std::size_t> &n_bins,
        const std::vector<std::pair<typename grid_t::loc_bin_index, dindex>>
            &bin_capacities = {},
        const std::vector<std::vector<scalar_type>> &ax_bin_edges =
            std::vector<std::vector<scalar_type>>()) {

        m_grid = m_factory.template new_grid<grid_t>(
            spans, n_bins, bin_capacities, ax_bin_edges);
    }

    /// Fill grid from existing volume using a bin filling strategy
    /// This can also be called without a volume builder
    template <typename volume_type, typename... Args>
    DETRAY_HOST void fill_grid(
        const detector_t &det, const volume_type &vol,
        const typename detector_t::geometry_context ctx = {},
        const bin_filler_t bin_filler = {}, Args &&... args) {

        bin_filler(m_grid, det, vol, ctx, args...);
    }

    /// Fill grid from externally provided surfaces - temporary solution until
    /// the volume builders can be deployed in the toy detector
    template <typename volume_type, typename surface_container_t,
              typename transform_container_t, typename mask_container_t,
              typename... Args>
    DETRAY_HOST void fill_grid(
        const volume_type &vol, const surface_container_t &surfaces,
        const transform_container_t &transforms, const mask_container_t &masks,
        const typename detector_t::geometry_context ctx = {},
        const bin_filler_t bin_filler = {}, Args &&... args) {

        bin_filler(m_grid, vol, surfaces, transforms, masks, ctx, args...);
    }

    /// Add the volume and the grid to the detector @param det
    DETRAY_HOST
    auto build(detector_t &det, typename detector_t::geometry_context ctx = {})
        -> typename detector_t::volume_type * override {

        using surface_desc_t = typename detector_t::surface_type;

        // Add the surfaces (portals and/or passives) that are owned by the vol
        typename detector_t::volume_type *vol_ptr =
            volume_decorator<detector_t>::build(det, ctx);

        // Find the surfaces that should be filled into the grid
        const auto vol = tracking_volume{det, vol_ptr->index()};

        // Grid has not been filled previously, fill it automatically
        if (m_grid.size() == 0u) {

            std::vector<surface_desc_t> surfaces{};
            for (auto &sf_desc : vol.surfaces()) {

                if (sf_desc.is_sensitive() ||
                    (m_add_passives && sf_desc.is_passive())) {
                    surfaces.push_back(sf_desc);
                }
            }

            this->fill_grid(
                tracking_volume{det,
                                volume_decorator<detector_t>::operator()()},
                surfaces, det.transform_store(), det.mask_store(), ctx);
        } else {
            // The grid is prefilled with surface descriptors that contain the
            // correct LOCAL surface indices per bin (e.g. from file IO).
            // Now add the rest of the linking information, which is only
            // available after the volume builder ran
            for (surface_desc_t &sf_desc : m_grid.all()) {

                assert(!detail::is_invalid_value(sf_desc.index()));

                dindex glob_idx{vol_ptr->to_global_sf_index(sf_desc.index())};
                const auto &new_sf_desc = det.surface(glob_idx);

                assert(new_sf_desc.index() == glob_idx);
                assert(!new_sf_desc.barcode().is_invalid());

                sf_desc = new_sf_desc;
            }
        }

        // Add the grid to the detector and link it to its volume
        constexpr auto gid{detector_t::accel::template get_id<grid_t>()};
        det._accelerators.template push_back<gid>(m_grid);
        vol_ptr->set_link(m_id, gid,
                          det.accelerator_store().template size<gid>() - 1);

        return vol_ptr;
    }

    /// @returns access to the new grid
    DETRAY_HOST
    auto &get() { return m_grid; }

    private:
    link_id_t m_id{link_id_t::e_sensitive};
    grid_factory_t m_factory{};
    typename grid_t::template type<true> m_grid{};
    bin_filler_t m_bin_filler{};
    bool m_add_passives{false};
};

/// Grid builder from single components
template <typename detector_t,
          template <class, template <std::size_t> class, typename>
          class grid_factory_t,
          typename grid_shape_t, typename bin_t,
          template <std::size_t> class serializer_t,
          axis::bounds e_bounds = axis::bounds::e_closed,
          typename algebra_t = typename detector_t::transform3,
          template <typename, typename> class... binning_ts>
using grid_builder_type = grid_builder<
    detector_t,
    typename grid_factory_t<bin_t, serializer_t, algebra_t>::template grid_type<
        axes<grid_shape_t, e_bounds, binning_ts...>>,
    grid_factory_t<bin_t, serializer_t, algebra_t>>;

}  // namespace detray
