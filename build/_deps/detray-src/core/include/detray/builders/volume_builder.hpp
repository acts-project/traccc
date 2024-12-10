/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/builders/surface_factory_interface.hpp"
#include "detray/builders/volume_builder_interface.hpp"
#include "detray/definitions/geometry.hpp"
#include "detray/geometry/tracking_surface.hpp"
#include "detray/utils/grid/detail/concepts.hpp"

// System include(s)
#include <algorithm>
#include <memory>
#include <string>

namespace detray {

namespace detail {

/// A functor to update the mask index in surface descriptors
struct mask_index_update;

}  // namespace detail

/// @brief Provides basic functionality to build detector volumes
template <typename detector_t>
class volume_builder : public volume_builder_interface<detector_t> {

    public:
    using scalar_type = typename detector_t::scalar_type;
    using volume_type = typename detector_t::volume_type;
    using geo_obj_ids = typename detector_t::geo_obj_ids;

    /// Parametrized Constructor
    ///
    /// @param id flags the type of volume geometry (e.g. cylindrical, cuboid)
    /// @param idx the index of the volume in the detector volume container
    volume_builder(const volume_id id, const dindex idx = 0) : m_volume{id} {

        m_volume.set_index(idx);
        m_volume.set_material(volume_type::material_id::e_none, 0u);

        // The first acceleration data structure in every volume is a brute
        // force method that will at least contain the portals
        m_volume.template set_accel_link<
            static_cast<typename volume_type::object_id>(0)>(
            detector_t::accel::id::e_default, 0);
    };

    /// Adds the @param name of the volume to a @param name_map
    template <typename name_map>
    DETRAY_HOST void add_name(const name_map& names, std::string&& name) {
        names.at(vol_index()) = std::move(name);
    }

    /// @returns the volume index in the detector volume container
    DETRAY_HOST
    auto vol_index() const -> dindex override { return m_volume.index(); }

    /// Toggles whether sensitive surfaces are added to the brute force method
    DETRAY_HOST
    void has_accel(bool toggle) override { m_has_accel = toggle; }

    /// @returns whether sensitive surfaces are added to the brute force method
    DETRAY_HOST
    bool has_accel() const override { return m_has_accel; }

    /// Access to the volume under construction - const
    DETRAY_HOST
    auto operator()() const -> const
        typename detector_t::volume_type& override {
        return m_volume;
    }

    /// Access to the volume under construction - non-const
    DETRAY_HOST
    auto operator()() -> typename detector_t::volume_type& override {
        return m_volume;
    }

    /// Build the volume with internal surfaces and portals and add it to the
    /// detector instance @param det
    DETRAY_HOST
    auto build(detector_t& det, typename detector_t::geometry_context ctx = {})
        -> typename detector_t::volume_type* override {
        // Prepare volume data
        m_volume.set_index(static_cast<dindex>(det.volumes().size()));

        m_volume.set_transform(det.transform_store().size());
        det._transforms.push_back(m_trf);

        // Add all data from the builder to the detector containers
        add_to_detector(ctx, det);

        // Reset after the data was added to the detector
        m_surfaces.clear();
        m_transforms.clear(ctx);
        m_masks.clear_all();

        // Pass to decorator builders
        return &(det._volumes.back());
    }

    /// Adds a placement transform @param trf for the volume
    DETRAY_HOST
    void add_volume_placement(
        const typename detector_t::transform3_type& trf = {}) override {
        m_trf = trf;
    }

    /// Constructs a placement transform with identity rotation and translation
    /// @param t for the volume
    DETRAY_HOST
    void add_volume_placement(
        const typename detector_t::point3_type& t) override {
        m_trf = typename detector_t::transform3_type{t};
    }

    /// Constructs a placement transform from axes @param x and @param z
    /// and the translation @param t for the volume
    DETRAY_HOST
    void add_volume_placement(
        const typename detector_t::point3_type& t,
        const typename detector_t::vector3_type& x,
        const typename detector_t::vector3_type& z) override {
        m_trf = typename detector_t::transform3_type{t, z, x, true};
    }

    /// Add data for (a) new surface(s) to the builder
    DETRAY_HOST
    void add_surfaces(
        std::shared_ptr<surface_factory_interface<detector_t>> sf_factory,
        typename detector_t::geometry_context ctx = {}) override {
        (*sf_factory)(m_volume, m_surfaces, m_transforms, m_masks, ctx);
    }

    protected:
    /// @returns Access to the surface descriptor data
    typename detector_t::surface_lookup_container& surfaces() override {
        return m_surfaces;
    }

    /// @returns Access to the surface/volume transform data
    typename detector_t::transform_container& transforms() override {
        return m_transforms;
    }

    /// @returns Access to the surface mask data
    typename detector_t::mask_container& masks() override { return m_masks; }

    /// Adds a new full set of volume components (e.g. transforms or masks)
    /// to the global detector data stores and updates all links.
    ///
    /// @param ctx is the geometry_context of the call
    /// @param det is the detector instance that the volume should be added to
    ///
    /// @note can throw an exception if input data is inconsistent
    template <geo_obj_ids surface_id = static_cast<geo_obj_ids>(0)>
    DETRAY_HOST auto add_to_detector(
        const typename detector_t::geometry_context ctx,
        detector_t& det) noexcept(false) -> void {

        // Append transforms
        const auto trf_offset = det.transform_store().size(ctx);
        det._transforms.append(std::move(m_transforms), ctx);

        // Surface index offset in the global detector container
        auto sf_offset{static_cast<dindex>(det.surfaces().size())};

        /// Find the surface range specifed by @param sf_id
        auto find_range = [&](auto sf_id) {
            // Compare given id to surface barcode
            auto is_sf_type = [sf_id](const auto& sf) {
                return sf.id() == sf_id;
            };

            auto first = static_cast<dindex>(
                math::abs(std::ranges::find_if(m_surfaces, is_sf_type) -
                          std::begin(m_surfaces)));

            auto last = static_cast<dindex>(
                math::abs(std::ranges::rend(m_surfaces) -
                          std::ranges::find_if(std::ranges::rbegin(m_surfaces),
                                               std::ranges::rend(m_surfaces),
                                               is_sf_type)));

            // Set correct empty range, otherwise shift by global surface offset
            return (first >= last)
                       ? dindex_range{}
                       : dindex_range{first + sf_offset, last + sf_offset};
        };

        m_volume.template update_sf_link<surface_id::e_portal>(
            find_range(surface_id::e_portal));

        m_volume.template update_sf_link<surface_id::e_sensitive>(
            find_range(surface_id::e_sensitive));

        m_volume.template update_sf_link<surface_id::e_passive>(
            find_range(surface_id::e_passive));

        // Update mask and transform index of surfaces and set the
        // correct index of the surface in container
        std::size_t n_portals{0u};
        for (auto& sf_desc : m_surfaces) {

            const auto sf = tracking_surface{det, sf_desc};

            sf.template visit_mask<detail::mask_index_update>(sf_desc);
            sf_desc.set_volume(m_volume.index());
            sf_desc.update_transform(trf_offset);
            sf_desc.set_index(sf_offset++);

            if (sf_desc.is_portal()) {
                ++n_portals;
            }

            det._surfaces.insert(sf_desc);
        }

        // Place the appropriate surfaces in the brute force search method.
        constexpr auto default_acc_id{detector_t::accel::id::e_default};

        // Strip the source link from the lookup data structure
        typename detector_t::surface_container descriptors;
        descriptors.reserve(m_surfaces.size());
        std::ranges::transform(
            m_surfaces, std::back_inserter(descriptors),
            [](typename detector_t::surface_lookup_container::value_type& sf) {
                return static_cast<
                    typename detector_t::surface_container::value_type>(sf);
            });

        // Add portals to brute force navigation method
        if (m_has_accel) {
            typename detector_t::surface_container portals{};
            portals.reserve(n_portals);

            std::ranges::copy_if(
                descriptors, std::back_inserter(portals),
                [](auto& sf_desc) { return !sf_desc.is_sensitive(); });

            // Add only the portals to the brute force method
            det._accelerators.template push_back<default_acc_id>(
                std::move(portals));
        } else {
            // Add all surfaces to the brute force method
            det._accelerators.template push_back<default_acc_id>(
                std::move(descriptors));
        }

        m_volume.template set_accel_link<surface_id>(
            default_acc_id,
            det.accelerator_store().template size<default_acc_id>() - 1u);

        // Append masks
        det._masks.append(std::move(m_masks));

        // Finally, add the volume descriptor to the detector
        det._volumes.push_back(m_volume);
    }

    private:
    /// Whether the volume will get an acceleration structure
    bool m_has_accel{false};

    /// Volume descriptor of the volume under construction
    typename detector_t::volume_type m_volume{};
    /// Placement of the volume under construction
    typename detector_t::transform3_type m_trf{};

    /// Data of contained surfaces
    /// @{
    typename detector_t::surface_lookup_container m_surfaces{};
    typename detector_t::transform_container m_transforms{};
    typename detector_t::mask_container m_masks{};
    /// @}
};

namespace detail {

/// A functor to update the mask index in surface objects
struct mask_index_update {

    template <typename group_t, typename index_t, typename surface_t>
    DETRAY_HOST inline void operator()(const group_t& group,
                                       const index_t& /*index*/,
                                       surface_t& sf) const {
        sf.update_mask(static_cast<dindex>(group.size()));
    }
};

/// TODO: Remove once the material builder is used everywhere
/// A functor to update the material index in surface objects
struct material_index_update {

    template <typename group_t, typename index_t, typename surface_t>
    DETRAY_HOST inline void operator()(
        [[maybe_unused]] const group_t& group,
        [[maybe_unused]] const index_t& /*index*/,
        [[maybe_unused]] surface_t& sf) const {
        if constexpr (!concepts::grid<typename group_t::value_type>) {
            sf.update_material(static_cast<dindex>(group.size()));
        }
    }
};

}  // namespace detail

}  // namespace detray
