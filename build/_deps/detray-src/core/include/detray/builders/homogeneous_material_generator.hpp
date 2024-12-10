/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/builders/surface_factory_interface.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/materials/material.hpp"
#include "detray/materials/predefined_materials.hpp"

// System include(s)
#include <sstream>
#include <stdexcept>

namespace detray {

/// @brief Configuration for the homogeneous material generator
template <typename scalar_t>
struct hom_material_config {
    /// Type of material to put on the passive surfaces
    material<scalar_t> m_passive_material{silicon<scalar_t>{}};
    /// Type of material to put on the sensitive surfaces
    material<scalar_t> m_sensitive_material{silicon<scalar_t>{}};
    /// Type of material to put on the portal surfaces
    material<scalar_t> m_portal_material{vacuum<scalar_t>{}};
    /// Minimal envelope for the portals (used in autofitting)
    scalar_t m_thickness{1.5f * unit<scalar_t>::mm};

    /// Setters
    /// @{
    constexpr hom_material_config &passive_material(
        const material<scalar_t> &m) {
        m_passive_material = m;
        return *this;
    }
    constexpr hom_material_config &sensitive_material(
        const material<scalar_t> &m) {
        m_sensitive_material = m;
        return *this;
    }
    constexpr hom_material_config &portal_material(
        const material<scalar_t> &m) {
        m_portal_material = m;
        return *this;
    }
    constexpr hom_material_config &thickness(const scalar_t t) {
        m_thickness = t;
        return *this;
    }
    /// @}

    /// Getters
    /// @{
    constexpr const material<scalar_t> &passive_material() const {
        return m_passive_material;
    }
    constexpr const material<scalar_t> &sensitive_material() const {
        return m_sensitive_material;
    }
    constexpr const material<scalar_t> &portal_material() const {
        return m_portal_material;
    }
    constexpr scalar_t thickness() const { return m_thickness; }
    /// @}
};

/// @brief Surface factory decorator that adds homogeneous material to surfaces
///
/// @tparam detector_t the type of detector the volume belongs to.
template <typename detector_t>
class homogeneous_material_generator final
    : public factory_decorator<detector_t> {

    using scalar_t = typename detector_t::scalar_type;

    public:
    /// Construct from configuration @param cfg
    DETRAY_HOST
    homogeneous_material_generator(
        std::unique_ptr<surface_factory_interface<detector_t>> factory,
        const hom_material_config<scalar_t> cfg)
        : factory_decorator<detector_t>(std::move(factory)), m_cfg{cfg} {}

    /// Call the underlying surface factory and record the surface range that
    /// was produced
    ///
    /// @param volume the volume the portals need to be added to.
    /// @param surfaces the surface collection to wrap and to add the portals to
    /// @param transforms the transforms of the surfaces.
    /// @param masks the masks of the surfaces.
    /// @param ctx the geometry context (not needed for portals).
    DETRAY_HOST
    auto operator()(typename detector_t::volume_type &volume,
                    typename detector_t::surface_lookup_container &surfaces,
                    typename detector_t::transform_container &transforms,
                    typename detector_t::mask_container &masks,
                    typename detector_t::geometry_context ctx = {})
        -> dindex_range override {

        auto [lower, upper] =
            (*this->get_factory())(volume, surfaces, transforms, masks, ctx);

        m_surface_range = {lower, upper};

        return {lower, upper};
    }

    /// Create material slabs or rods for all surfaces that the undelying
    /// surface factory builds.
    ///
    /// @param surfaces surface container of the volume builder that should get
    ///                 decorated with material.
    /// @param material material store of the volume builder that the new
    ///                 materials get added to.
    DETRAY_HOST
    auto operator()(typename detector_t::surface_lookup_container &surfaces,
                    typename detector_t::material_container &materials) {

        using material_id = typename detector_t::materials::id;
        using link_t = typename detector_t::surface_type::material_link;

        assert(surfaces.size() >= (m_surface_range[1] - m_surface_range[0]));

        // Add the material to the surfaces that the data links against
        for (auto &sf : detray::ranges::subrange(surfaces, m_surface_range)) {

            const material<scalar_t> *mat_ptr{nullptr};

            // Get the correct material for this surface type
            constexpr vacuum<scalar_t> vac{};
            switch (sf.id()) {
                case surface_id::e_passive: {
                    const auto &mat = m_cfg.passive_material();
                    mat_ptr = (mat != vac) ? &mat : nullptr;
                    break;
                }
                case surface_id::e_sensitive: {
                    const auto &mat = m_cfg.sensitive_material();
                    mat_ptr = (mat != vac) ? &mat : nullptr;
                    break;
                }
                case surface_id::e_portal: {
                    const auto &mat = m_cfg.portal_material();
                    mat_ptr = (mat != vac) ? &mat : nullptr;
                    break;
                }
                case surface_id::e_unknown: {
                    std::stringstream err_stream{};
                    err_stream << sf << std::endl;
                    throw std::runtime_error(
                        "Encountered surface of unknown type during material "
                        "generation: " +
                        err_stream.str());
                    break;
                }
                default: {
                    break;
                }
            }

            // Found suitable material for this surface?
            if (mat_ptr == nullptr) {
                continue;
            }

            // Handle line shaped surfaces differently
            bool is_line{false};
            link_t mat_link;

            if constexpr (detector_t::materials::template is_defined<
                              material_rod<scalar_t>>()) {

                using mask_id = typename detector_t::masks::id;

                // If the current surface is a line, generate a material rod
                const mask_id sf_mask_id = sf.mask().id();
                if (sf_mask_id == mask_id::e_straw_tube ||
                    sf_mask_id == mask_id::e_drift_cell) {

                    is_line = true;

                    auto &mat_coll =
                        materials.template get<material_id::e_rod>();
                    mat_coll.emplace_back(*mat_ptr, m_cfg.thickness());

                    mat_link = {material_id::e_rod,
                                static_cast<dindex>(mat_coll.size() - 1u)};
                }
            }

            // For all surfaces that are not lines, generate a material slab
            if (!is_line) {
                auto &mat_coll = materials.template get<material_id::e_slab>();
                mat_coll.emplace_back(*mat_ptr, m_cfg.thickness());

                mat_link = {material_id::e_slab,
                            static_cast<dindex>(mat_coll.size() - 1u)};
            }

            // Set the initial surface material link (will be updated when
            // added to the detector)
            sf.material() = mat_link;
        }
    }

    private:
    /// Material generator configuration
    hom_material_config<scalar_t> m_cfg;
    /// Range of surface indices for which to generate material
    dindex_range m_surface_range{};
};

}  // namespace detray
