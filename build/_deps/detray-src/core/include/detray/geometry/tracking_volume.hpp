/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/containers.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/geometry.hpp"
#include "detray/geometry/detail/volume_kernels.hpp"
#include "detray/materials/material.hpp"
#include "detray/utils/ranges.hpp"

// System include(s)
#include <algorithm>
#include <iostream>
#include <sstream>

namespace detray {

/// @brief Facade for a detray detector volume.
///
/// Volume class that acts as a logical container in the detector for geometry
/// objects, i.e. surfaces. The volume boundary surfaces, the so-called portals,
/// carry index links that join adjacent volumes. The volume class itself does
/// not contain any data itself, but keeps a descriptor with index-based links
/// into the data containers that are managed by the detector.
/// Every type of surface that is known by the volume (determined by the type
/// and size of the ID enum in the descriptor) lives in it's own geometry
/// accelerator data structure, e.g. portals reside in a brute force
/// accelerator (a simple vector), while sensitive surfaces are usually sorted
/// into a spacial grid.
template <typename detector_t>  // @TODO: This needs a concept
class tracking_volume {

    /// Scalar type
    using scalar_type = typename detector_t::scalar_type;
    using point3_type = typename detector_t::point3_type;

    /// Volume descriptor type
    using descr_t = typename detector_t::volume_type;
    using context_t = typename detector_t::geometry_context;

    public:
    /// In case the geometry needs to be printed
    using name_map = dmap<dindex, std::string>;

    /// Not allowed: always needs a detector and a descriptor.
    tracking_volume() = delete;

    /// Constructor from detector @param det and volume descriptor
    /// @param vol_idx from that detector.
    constexpr tracking_volume(const detector_t &det, const descr_t &desc)
        : m_detector{det}, m_desc{desc} {}

    /// Constructor from detector @param det and volume index @param vol_idx in
    /// that detector.
    constexpr tracking_volume(const detector_t &det, const dindex vol_idx)
        : tracking_volume(det, det.volume(vol_idx)) {}

    /// @returns access to the underlying detector
    DETRAY_HOST_DEVICE
    auto detector() const -> const detector_t & { return m_detector; }

    /// Equality operator
    ///
    /// @param rhs is the right hand side to be compared to
    DETRAY_HOST_DEVICE
    constexpr auto operator==(const tracking_volume &rhs) const -> bool {
        return (&m_detector == &(rhs.m_detector) && m_desc == rhs.m_desc);
    }

    /// @returns the volume shape id, e.g. 'cylinder'.
    DETRAY_HOST_DEVICE
    constexpr auto id() const -> volume_id { return m_desc.id(); }

    /// @returns the index of the volume in the detector volume container.
    DETRAY_HOST_DEVICE
    constexpr auto index() const -> dindex { return m_desc.index(); }

    /// @returns the (non contextual) transform for the placement of the
    /// volume in the detector geometry.
    DETRAY_HOST_DEVICE
    constexpr auto transform() const -> const
        typename detector_t::transform3_type & {
        return m_detector.transform_store().at(m_desc.transform());
    }

    /// @returns the center point of the volume.
    DETRAY_HOST_DEVICE
    constexpr auto center() const -> point3_type {
        return transform().translation();
    }

    /// @returns a pointer to the material parameters at the local position
    /// @param loc_p
    DETRAY_HOST_DEVICE constexpr auto material_parameters(
        const point3_type &loc_p) const
        -> const detray::material<scalar_type> * {
        return visit_material<typename detail::get_material_params>(loc_p);
    }

    /// @returns true if the volume carries material.
    DETRAY_HOST_DEVICE
    constexpr auto has_material() const -> bool {
        return m_desc.material().id() !=
               static_cast<typename descr_t::material_link::id_type>(
                   detector_t::materials::id::e_none);
    }

    /// @returns an iterator pair for the requested type of surfaces.
    template <surface_id sf_type = surface_id::e_all>
    DETRAY_HOST_DEVICE constexpr decltype(auto) surfaces() const {
        if constexpr (sf_type == surface_id::e_all) {
            return detray::ranges::subrange{m_detector.surfaces(),
                                            m_desc.full_sf_range()};
        } else {
            return detray::ranges::subrange{m_detector.surfaces(),
                                            m_desc.template sf_link<sf_type>()};
        }
    }

    /// @returns an iterator pair for the requested type of surfaces.
    DETRAY_HOST_DEVICE constexpr decltype(auto) portals() const {
        return detray::ranges::subrange{
            m_detector.surfaces(),
            m_desc.template sf_link<surface_id::e_portal>()};
    }

    /// Apply a functor to all surfaces in the volume's acceleration structures
    ///
    /// @tparam functor_t the prescription to be applied to the surfaces
    /// @tparam Args      types of additional arguments to the functor
    template <typename functor_t,
              int I = static_cast<int>(descr_t::object_id::e_size) - 1,
              typename... Args>
    DETRAY_HOST_DEVICE constexpr void visit_surfaces(Args &&... args) const {
        visit_surfaces_impl<detail::surface_getter<functor_t>>(
            std::forward<Args>(args)...);
    }

    /// Apply a functor to a neighborhood of surfaces around a track position
    /// in the volume.
    ///
    /// @tparam functor_t the prescription to be applied to the surfaces (
    ///                   customization point for the navigation)
    /// @tparam track_t   the track around which to build up the neighborhood
    /// @tparam Args      types of additional arguments to the functor
    template <typename functor_t,
              int I = static_cast<int>(descr_t::object_id::e_size) - 1,
              typename track_t, typename config_t, typename... Args>
    DETRAY_HOST_DEVICE constexpr void visit_neighborhood(
        const track_t &track, const config_t &cfg, const context_t &ctx,
        Args &&... args) const {
        visit_surfaces_impl<detail::neighborhood_getter<functor_t>>(
            m_detector, m_desc, track, cfg, ctx, std::forward<Args>(args)...);
    }

    /// Call a functor on the volume material with additional arguments.
    ///
    /// @tparam functor_t the prescription to be applied to the material
    /// @tparam Args      types of additional arguments to the functor
    template <typename functor_t, typename... Args>
    DETRAY_HOST_DEVICE constexpr auto visit_material(Args &&... args) const {
        const auto &materials = m_detector.material_store();

        return materials.template visit<functor_t>(m_desc.material(),
                                                   std::forward<Args>(args)...);
    }

    /// Do a consistency check on the volume after building the detector.
    ///
    /// @param os output stream for error messages.
    ///
    /// @returns true if the volume is consistent
    DETRAY_HOST bool self_check(std::ostream &os) const {
        if (id() == volume_id::e_unknown) {
            os << "ERROR: Unknown volume shape type in volume:\n"
               << *this << std::endl;
            return false;
        }
        if (detail::is_invalid_value(index())) {
            os << "ERROR: Volume index undefined in volume:\n"
               << *this << std::endl;
            return false;
        }
        if (index() >= m_detector.volumes().size()) {
            os << "ERROR: Volume index out of bounds in volume:\n"
               << *this << std::endl;
            return false;
        }
        if (detail::is_invalid_value(m_desc.transform())) {
            os << "ERROR: Volume transform undefined in volume:\n"
               << *this << std::endl;
            return false;
        }
        if (m_desc.transform() >= m_detector.transform_store().size()) {
            os << "ERROR: Volume transform index out of bounds in volume:\n"
               << *this << std::endl;
            return false;
        }
        // Only check, if there is material in the detector
        if (!m_detector.material_store().all_empty() && has_material() &&
            m_desc.material().is_invalid_index()) {
            os << "ERROR: Volume does not have valid material link:\n"
               << *this << std::endl;
            return false;
        }
        const auto &acc_link = m_desc.accel_link();
        if (detail::is_invalid_value(acc_link[0])) {
            os << "ERROR: Link to portal lookup broken: " << acc_link[0]
               << "\n in volume: " << *this << std::endl;
            return false;
        }
        if (const auto &pt_link =
                m_desc.template sf_link<surface_id::e_portal>();
            detail::is_invalid_value(pt_link)) {
            os << "ERROR: Link to portal surfaces broken: " << pt_link
               << "\n in volume: " << *this << std::endl;
            return false;
        }
        // Check consistency of surface ranges
        std::vector<dindex_range> sf_ranges = {
            m_desc.template sf_link<surface_id::e_portal>()};

        // Only add the other ranges in case they are not empty
        if (const auto &sens_range =
                m_desc.template sf_link<surface_id::e_sensitive>();
            sens_range[0] != sens_range[1]) {
            sf_ranges.push_back(sens_range);
        }
        if (const auto &psv_range =
                m_desc.template sf_link<surface_id::e_passive>();
            psv_range[0] != psv_range[1]) {
            sf_ranges.push_back(psv_range);
        }

        // Sort and check that the ranges are contiguous
        auto compare_ranges = [](const dindex_range &rg1,
                                 const dindex_range &rg2) {
            return rg1[0] < rg2[0];
        };

        std::ranges::sort(sf_ranges, compare_ranges);

        if ((sf_ranges.size() > 1 && sf_ranges[0][1] != sf_ranges[1][0]) ||
            (sf_ranges.size() > 2 && sf_ranges[1][1] != sf_ranges[2][0])) {
            os << "ERROR: Surface index ranges not contigous: "
               << m_desc.sf_link() << "\n in volume: " << *this << std::endl;
            return false;
        }

        // Warnings
        bool suspicious_links = false;
        std::stringstream warnigns{};
        for (std::size_t i = 1u; i < acc_link.size(); ++i) {
            // An acceleration data structure link was set, but is invalid
            if (!acc_link[i].is_invalid_id() &&
                acc_link[i].is_invalid_index()) {
                suspicious_links = true;
                warnigns << "Link to acceleration data structure "
                         << static_cast<int>(acc_link[i].id()) << " is invalid"
                         << std::endl;
            }
        }
        if (suspicious_links) {
            std::cout << "WARNING: " << warnigns.str()
                      << " in volume: " << *this << std::endl;
        }

        return true;
    }

    /// @returns the volume name (add an offset for the detector name).
    DETRAY_HOST_DEVICE
    constexpr auto name(const name_map &names) const -> const std::string & {
        return names.at(m_desc.index() + 1u);
    }

    /// @returns a string stream that prints the volume details
    DETRAY_HOST
    friend std::ostream &operator<<(std::ostream &os,
                                    const tracking_volume &v) {
        os << "id: " << static_cast<int>(v.m_desc.id());
        os << " | index: " << v.m_desc.index();
        os << " | trf.: " << v.m_desc.transform();
        os << " | acc link: " << v.m_desc.accel_link();
        os << " | sf link: " << v.m_desc.sf_link();
        os << " | mat link: " << v.m_desc.material();

        return os;
    }

    private:
    /// Apply a functor to all acceleration structures of this volume.
    ///
    /// @tparam functor_t the prescription to be applied to the acc structure
    /// @tparam Args      types of additional arguments to the functor
    template <typename functor_t,
              int I = static_cast<int>(descr_t::object_id::e_size) - 1,
              typename... Args>
    DETRAY_HOST_DEVICE constexpr void visit_surfaces_impl(
        Args &&... args) const {
        // Get the acceleration data structures for this volume and only visit,
        // if object type is contained in volume
        if (const auto &link{m_desc.template accel_link<
                static_cast<typename descr_t::object_id>(I)>()};
            !link.is_invalid()) {
            // Run over the surfaces in a single acceleration data structure
            // and apply the functor to the resulting neighborhood
            m_detector.accelerator_store().template visit<functor_t>(
                link, std::forward<Args>(args)...);
        }
        // Check the next surface type
        if constexpr (I > 0) {
            visit_surfaces_impl<functor_t, I - 1, Args...>(
                std::forward<Args>(args)...);
        }
    }

    /// Access to the detector stores
    const detector_t &m_detector;
    /// Access to the descriptor
    const descr_t &m_desc;
};

}  // namespace detray
