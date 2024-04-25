/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/track_parameters.hpp"

// detray include(s).
#include "detray/geometry/barcode.hpp"
#include "detray/geometry/surface.hpp"
#include "detray/navigation/intersection/ray_intersector.hpp"
#include "detray/navigation/intersection_kernel.hpp"
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
    using cxt_t = typename detector_t::geometry_context;

    /// Constructor with detector
    ///
    /// @param det input detector
    /// @param stddevs standard deviations for parameter smearing
    seed_generator(const detector_t& det,
                   const std::array<scalar, e_bound_size>& stddevs,
                   const std::size_t sd = 0)
        : m_detector(det), m_stddevs(stddevs) {
        generator.seed(sd);
    }

    /// Seed generator operation
    ///
    /// @param vertex vertex of particle
    /// @param stddevs standard deviations for track parameter smearing
    bound_track_parameters operator()(
        const detray::geometry::barcode surface_link,
        const free_track_parameters& free_param) {

        // Get bound parameter
        const detray::surface<detector_t> sf{m_detector, surface_link};

        const cxt_t ctx{};
        auto bound_vec = sf.free_to_bound_vector(ctx, free_param.vector());

        auto bound_cov =
            matrix_operator().template zero<e_bound_size, e_bound_size>();

        bound_track_parameters bound_param{surface_link, bound_vec, bound_cov};

        // Type definitions
        using transform3_type = typename detector_t::transform3;
        using scalar_type = typename detector_t::scalar_type;
        using intersection_type =
            detray::intersection2D<typename detector_t::surface_type,
                                   transform3_type>;
        using interactor_type =
            detray::pointwise_material_interactor<transform3_type>;

        intersection_type sfi;
        sfi.sf_desc = m_detector.surface(surface_link);
        sf.template visit_mask<
            detray::intersection_update<detray::ray_intersector>>(
            detray::detail::ray<transform3_type>(free_param.vector()), sfi,
            m_detector.transform_store(),
            sf.is_portal() ? 0.f : 50.f * unit<scalar_type>::um,
            -100.f * unit<scalar_type>::um);

        if (!(std::abs(std::abs(sf.cos_angle(ctx, bound_param.dir(),
                                             bound_param.bound_local())) -
                       sfi.cos_incidence_angle) < 0.0001f)) {
            std::cout << "seed gen" << std::endl;
            std::cout << m_detector.surface(surface_link) << std::endl;
            std::cout << sfi << std::endl;
            std::cout << sf.cos_angle(ctx, bound_param.dir(),
                                      bound_param.bound_local())
                      << ", " << sfi.cos_incidence_angle << std::endl;
        }

        // Apply interactor
        typename interactor_type::state interactor_state;
        interactor_state.do_multiple_scattering = false;
        interactor_type{}.update(
            bound_param, interactor_state,
            static_cast<int>(detray::navigation::direction::e_backward), sf,
            sfi.cos_incidence_angle);

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

    // Detector object
    const detector_t& m_detector;
    /// Standard deviations for parameter smearing
    std::array<scalar, e_bound_size> m_stddevs;
};

}  // namespace traccc
