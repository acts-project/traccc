/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/builders/cuboid_portal_generator.hpp"
#include "detray/builders/detector_builder.hpp"
#include "detray/builders/homogeneous_material_builder.hpp"
#include "detray/builders/homogeneous_material_generator.hpp"
#include "detray/builders/homogeneous_volume_material_builder.hpp"
#include "detray/core/detector.hpp"
#include "detray/definitions/units.hpp"
#include "detray/detectors/telescope_metadata.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/materials/predefined_materials.hpp"
#include "detray/navigation/detail/trajectories.hpp"
#include "detray/utils/consistency_checker.hpp"

// Detray test include(s)
#include "detray/test/utils/detectors/factories/telescope_generator.hpp"

// Vecmem include(s)
#include <vecmem/memory/memory_resource.hpp>

// System include(s)
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

namespace detray {

namespace {

/// Configure the toy detector
template <typename mask_shape_t = rectangle2D,
          typename trajectory_t = detail::ray<ALGEBRA_PLUGIN<detray::scalar>>>
struct tel_det_config {

    /// Construct from existing mask
    tel_det_config(const mask<mask_shape_t> &m, const trajectory_t &t = {})
        : m_mask(m), m_trajectory(t) {
        // Configure the material generation
        m_material_config.sensitive_material(silicon_tml<scalar>())
            .passive_material(vacuum<scalar>())
            .portal_material(vacuum<scalar>())
            .thickness(80.f * unit<scalar>::um);
    }

    /// Construct from from mask parameter vector
    tel_det_config(std::vector<scalar> params, const trajectory_t &t = {})
        : tel_det_config(mask<mask_shape_t>{std::move(params), 0u}, t) {}

    /// Construct from mask parameters (except volume link, which is not needed)
    template <typename... Args>
    requires(std::is_same_v<Args, scalar> &&...) explicit tel_det_config(
        Args &&... args)
        : tel_det_config(mask<mask_shape_t>{0u, std::forward<Args>(args)...}) {}

    /// Mask of the test surfaces
    mask<mask_shape_t> m_mask;
    /// No. of test surfaces in the telescope
    unsigned int m_n_surfaces{10u};
    /// Length of the telescope
    scalar m_length{500.f * unit<scalar>::mm};
    /// Concrete positions where to place the surfaces along the pilot track
    std::vector<scalar> m_positions{};
    /// Configuration for the homogeneous material generator
    hom_material_config<scalar> m_material_config{};
    /// Material for volume
    material<scalar> m_volume_material = vacuum<scalar>();
    /// Pilot track along which to place the surfaces
    trajectory_t m_trajectory{};
    /// Safety envelope between the test surfaces and the portals
    scalar m_envelope{0.1f * unit<scalar>::mm};
    /// Run detector consistency check after reading
    bool m_do_check{true};

    /// Setters
    /// @{
    constexpr tel_det_config &module_mask(const mask<mask_shape_t> &m) {
        m_mask = m;
        return *this;
    }
    constexpr tel_det_config &n_surfaces(const unsigned int n) {
        m_n_surfaces = n;
        return *this;
    }
    constexpr tel_det_config &length(const scalar l) {
        assert((l > 0.f) &&
               "Telescope detector length must be greater than zero");
        m_length = l;
        return *this;
    }
    constexpr tel_det_config &positions(const std::vector<scalar> &dists) {
        m_positions.clear();
        std::ranges::copy_if(dists, std::back_inserter(m_positions),
                             [](scalar d) { return (d >= 0.f); });
        return *this;
    }
    constexpr tel_det_config &module_material(const material<scalar> &mat) {
        m_material_config.sensitive_material(mat);
        return *this;
    }
    constexpr tel_det_config &volume_material(const material<scalar> &mat) {
        m_volume_material = mat;
        return *this;
    }
    constexpr tel_det_config &mat_thickness(const scalar t) {
        assert(t >= 0.f && "Material thickness must be non-negative");
        m_material_config.thickness(t);
        return *this;
    }
    constexpr tel_det_config &pilot_track(const trajectory_t &traj) {
        m_trajectory = traj;
        return *this;
    }
    constexpr tel_det_config &envelope(const scalar e) {
        assert(e > 0.f && "Portal envelope must be greater than zero");
        m_envelope = e;
        return *this;
    }
    tel_det_config &do_check(const bool check) {
        m_do_check = check;
        return *this;
    }
    /// @}

    /// Getters
    /// @{
    constexpr const mask<mask_shape_t> &module_mask() const { return m_mask; }
    constexpr unsigned int n_surfaces() const { return m_n_surfaces; }
    constexpr scalar length() const { return m_length; }
    const std::vector<scalar> &positions() const { return m_positions; }
    constexpr const auto &material_config() const { return m_material_config; }
    constexpr auto &material_config() { return m_material_config; }
    constexpr const material<scalar> &module_material() const {
        return m_material_config.sensitive_material();
    }
    constexpr const material<scalar> &volume_material() const {
        return m_volume_material;
    }
    constexpr scalar mat_thickness() const {
        return m_material_config.thickness();
    }
    const trajectory_t &pilot_track() const { return m_trajectory; }
    constexpr scalar envelope() const { return m_envelope; }
    bool do_check() const { return m_do_check; }
    /// @}
};

}  // namespace

/// Builds a detray geometry that contains only one volume with one type of
/// surfaces. The detector is auto-constructed by following a trajectory
/// through space, along which the surfaces are placed. The portals are built
/// from the bounding box around the sensors.
///
/// @tparam mask_shape_t the type of mask for the telescope surfaces
/// @tparam trajectory_t the type of the pilot trajectory (ray/helix)
///
/// @param resource the memory resource for the detector containers
/// @param cfg configuration struct of the telescope detector
///
/// @returns a complete detector object
template <typename mask_shape_t = rectangle2D,
          typename trajectory_t = detail::ray<ALGEBRA_PLUGIN<detray::scalar>>>
inline auto build_telescope_detector(
    vecmem::memory_resource &resource,
    const tel_det_config<mask_shape_t, trajectory_t> &cfg =
        tel_det_config<mask_shape_t, trajectory_t>{20.f * unit<scalar>::mm,
                                                   20.f * unit<scalar>::mm}) {

    using builder_t =
        detector_builder<telescope_metadata<mask_shape_t>, volume_builder>;
    using detector_t = typename builder_t::detector_type;

    // Detector and volume names
    typename detector_t::name_map name_map = {{0u, "telescope_detector"},
                                              {1u, "telescope_world_0"}};

    builder_t det_builder;

    // Create an empty cuboid volume
    auto v_builder = det_builder.new_volume(volume_id::e_cuboid);

    // Identity transform (volume is centered at origin)
    v_builder->add_volume_placement();

    // Add module surfaces to volume
    using telescope_factory =
        telescope_generator<detector_t, mask_shape_t, trajectory_t>;
    std::unique_ptr<surface_factory_interface<detector_t>> tel_generator;

    if (cfg.positions().empty()) {
        // Automatically position the modules along pilot track
        tel_generator = std::make_unique<telescope_factory>(
            cfg.length(), cfg.n_surfaces(), cfg.module_mask().values(),
            cfg.pilot_track());
    } else {
        // Put the modules in the requested poritions along pilot track
        tel_generator = std::make_unique<telescope_factory>(
            cfg.positions(), cfg.module_mask().values(), cfg.pilot_track());
    }

    // Add homogeneous material description if a valid material was configured
    volume_builder_interface<detector_t> *vm_builder{v_builder};
    std::shared_ptr<surface_factory_interface<detector_t>> module_generator;

    if (cfg.module_material() != detray::vacuum<scalar>{}) {

        // Decorate the builders with homogeneous material
        vm_builder =
            det_builder
                .template decorate<homogeneous_material_builder<detector_t>>(
                    v_builder);

        if (!vm_builder) {
            throw std::runtime_error("Surface material decoration failed");
        }

        auto tel_mat_generator =
            std::make_shared<homogeneous_material_generator<detector_t>>(
                std::move(tel_generator), cfg.material_config());

        module_generator = std::move(tel_mat_generator);

    } else {
        module_generator = std::move(tel_generator);
    }

    // Add a bounding box of portals to the cuboid volume
    auto portal_generator =
        std::make_shared<cuboid_portal_generator<detector_t>>(cfg.envelope());

    vm_builder->add_surfaces(module_generator);
    // (!) The portals must be added after the modules to fit them correctly
    vm_builder->add_surfaces(portal_generator);

    det_builder.set_volume_finder(resource);
    det_builder.volume_finder().push_back(
        std::vector<dindex>{vm_builder->vol_index()});

    // If requested, add homogeneous volume material
    if (cfg.volume_material() != detray::vacuum<scalar>{}) {
        auto full_v_builder = det_builder.template decorate<
            homogeneous_volume_material_builder<detector_t>>(vm_builder);

        if (full_v_builder) {
            full_v_builder->set_material(cfg.volume_material());
        } else {
            throw std::runtime_error("Volume material decoration failed");
        }
    }

    // Build and return the detector
    auto det = det_builder.build(resource);

    if (cfg.do_check()) {
        const bool verbose_check{false};
        detray::detail::check_consistency(det, verbose_check, name_map);
    }

    return std::make_pair(std::move(det), std::move(name_map));
}

}  // namespace detray
