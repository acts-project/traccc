/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/track_parameters.hpp"

// detray include(s).
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/base_actor.hpp"
#include "detray/propagator/propagator.hpp"

// System include(s).
#include <random>

namespace traccc {

/// Seed track parameter generator
template <typename detector_t>
struct seed_generator {
    using matrix_operator = typename transform3::matrix_actor;

    /// Constructor with detector
    ///
    /// @param det input detector
    /// @param stddevs standard deviations for parameter smearing
    seed_generator(const detector_t& det,
                   const std::array<scalar, e_bound_size>& stddevs,
                   const std::size_t sd = 0)
        : m_detector(std::make_unique<detector_t>(det)), m_stddevs(stddevs) {
        generator.seed(sd);
    }

    /// Seed generator operation
    ///
    /// @param vertex vertex of particle
    /// @param stddevs standard deviations for track parameter smearing
    bound_track_parameters operator()(const geometry_id surface_link,
                                      const free_track_parameters& free_param) {

        // Get bound parameter
        auto bound_vec = m_detector->free_to_bound_vector(
            detray::geometry::barcode{surface_link}, free_param.vector());
        auto bound_cov =
            matrix_operator().template zero<e_bound_size, e_bound_size>();

        bound_track_parameters bound_param{
            detray::geometry::barcode{surface_link}, bound_vec, bound_cov};

        // Type definitions
        using transform3_type = typename detector_t::transform3;
        using intersection_type =
            detray::intersection2D<typename detector_t::surface_type,
                                   transform3_type>;
        using interactor_type =
            detray::pointwise_material_interactor<transform3_type>;

        const auto& mask_store = m_detector->mask_store();

        intersection_type sfi;
        sfi.surface =
            m_detector->surfaces(detray::geometry::barcode{surface_link});

        mask_store.template visit<detray::intersection_update>(
            sfi.surface.mask(),
            detray::detail::ray<transform3_type>(free_param.vector()), sfi,
            m_detector->transform_store());

        // Apply interactor
        typename interactor_type::state interactor_state;
        interactor_state.do_multiple_scattering = false;
        interactor_type{}.update(
            bound_param, interactor_state,
            static_cast<int>(detray::navigation::direction::e_backward), sfi,
            m_detector->material_store());

        for (std::size_t i = 0; i < e_bound_size; i++) {

            matrix_operator().element(bound_param.vector(), i, 0) =
                std::normal_distribution<scalar>(
                    matrix_operator().element(bound_param.vector(), i, 0),
                    m_stddevs[i])(generator);

            matrix_operator().element(bound_param.covariance(), i, i) =
                m_stddevs[i] * m_stddevs[i];
        }

        return bound_param;
    }

    private:
    // Random generator
    std::random_device rd{};
    std::mt19937 generator{rd()};

    /// Detector objects
    std::unique_ptr<detector_t> m_detector;
    /// Standard deviations for parameter smearing
    std::array<scalar, e_bound_size> m_stddevs;
};

}  // namespace traccc