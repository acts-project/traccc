/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/detail/chi2_cdf.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_smoother.hpp"
#include "traccc/fitting/kalman_filter/kalman_actor.hpp"
#include "traccc/fitting/kalman_filter/statistics_updater.hpp"

// detray include(s).
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/propagator.hpp"

// System include(s).
#include <limits>

namespace traccc {

/// Kalman fitter algorithm to fit a single track
template <typename stepper_t, typename navigator_t>
class kalman_fitter {

    public:
    // scalar type
    using scalar_type = typename stepper_t::scalar_type;

    // vector type
    template <typename T>
    using vector_type = typename navigator_t::template vector_type<T>;

    // navigator candidate type
    using intersection_type = typename navigator_t::intersection_type;

    /// Configuration type
    using config_type = fitting_config<scalar_type>;

    // transform3 type
    using transform3_type = typename stepper_t::transform3_type;

    // Detector type
    using detector_type = typename navigator_t::detector_type;

    // Field type
    using bfield_type = typename stepper_t::magnetic_field_type;

    // Actor types
    using aborter = detray::pathlimit_aborter;
    using transporter = detray::parameter_transporter<transform3_type>;
    using interactor = detray::pointwise_material_interactor<transform3_type>;
    using fit_actor = traccc::kalman_actor<transform3_type, vector_type>;
    using resetter = detray::parameter_resetter<transform3_type>;

    using actor_chain_type =
        detray::actor_chain<std::tuple, aborter, transporter, interactor,
                            fit_actor, resetter>;

    // Propagator type
    using propagator_type =
        detray::propagator<stepper_t, navigator_t, actor_chain_type>;

    /// Constructor with a detector
    ///
    /// @param det the detector object
    TRACCC_HOST_DEVICE
    kalman_fitter(const detector_type& det, const bfield_type& field,
                  const config_type& cfg)
        : m_detector(det), m_field(field), m_cfg(cfg) {}

    /// Kalman fitter state
    struct state {

        /// State constructor
        ///
        /// @param track_states the vector of track states
        TRACCC_HOST_DEVICE
        state(vector_type<track_state<transform3_type>>&& track_states)
            : m_fit_actor_state(std::move(track_states)) {}

        /// State constructor
        ///
        /// @param track_states the vector of track states
        TRACCC_HOST_DEVICE
        state(const vector_type<track_state<transform3_type>>& track_states)
            : m_fit_actor_state(track_states) {}

        /// @return the actor chain state
        TRACCC_HOST_DEVICE
        typename actor_chain_type::state operator()() {
            return std::tie(m_aborter_state, m_transporter_state,
                            m_interactor_state, m_fit_actor_state,
                            m_resetter_state);
        }

        /// Individual actor states
        typename aborter::state m_aborter_state{};
        typename transporter::state m_transporter_state{};
        typename interactor::state m_interactor_state{};
        typename fit_actor::state m_fit_actor_state;
        typename resetter::state m_resetter_state{};

        /// Fitting result per track
        fitter_info<transform3_type> m_fit_info;
    };

    /// Run the kalman fitter for a given number of iterations
    ///
    /// @tparam seed_parameters_t the type of seed track parameter
    ///
    /// @param seed_params seed track parameter
    /// @param fitter_state the state of kalman fitter
    template <typename seed_parameters_t>
    TRACCC_HOST_DEVICE void fit(
        const seed_parameters_t& seed_params, state& fitter_state,
        vector_type<intersection_type>&& nav_candidates = {}) {

        // Run the kalman filtering for a given number of iterations
        for (std::size_t i_it = 0; i_it < m_cfg.n_iterations; i_it++) {

            // Reset the iterator of kalman actor
            fitter_state.m_fit_actor_state.reset();

            if (i_it == 0) {
                filter(seed_params, fitter_state, std::move(nav_candidates));
            }
            // From the second iteration, seed parameter is the smoothed track
            // parameter at the first surface
            else if (i_it > 0) {

                auto& smoothed =
                    fitter_state.m_fit_actor_state.m_track_states[0].smoothed();

                const auto& mask_store = m_detector.mask_store();

                // Get intersection on surface
                intersection_type sfi;
                sfi.surface = m_detector.surfaces(
                    detray::geometry::barcode{smoothed.surface_link()});

                // Get free vector on surface
                auto free_vec = m_detector.bound_to_free_vector(
                    detray::geometry::barcode{smoothed.surface_link()},
                    smoothed.vector());

                mask_store.template visit<detray::intersection_update>(
                    sfi.surface.mask(),
                    detray::detail::ray<transform3_type>(free_vec), sfi,
                    m_detector.transform_store());

                // Apply material interaction backwardly to track state
                typename interactor::state interactor_state;
                interactor_state.do_multiple_scattering = false;
                interactor{}.update(
                    smoothed, interactor_state,
                    static_cast<int>(detray::navigation::direction::e_backward),
                    sfi, m_detector.material_store());

                // Make new seed parameter
                auto new_seed_params =
                    fitter_state.m_fit_actor_state.m_track_states[0].smoothed();

                // inflate cov
                /*
                auto& new_cov = new_seed_params.covariance();

                for (std::size_t i = 0; i < e_bound_size; i++) {
                    for (std::size_t j = 0; j < e_bound_size; j++) {
                        if (i == j && i != e_bound_qoverp) {
                            getter::element(new_cov, i, j) =
                                getter::element(new_cov, i, j) * 100.f;
                        }
                        if (i != j) {
                            getter::element(new_cov, i, j) = 0.f;
                        }
                    }
                }
                */
                filter(new_seed_params, fitter_state,
                       std::move(nav_candidates));
            }
        }
    }

    /// Run the kalman fitter for an iteration
    ///
    /// @tparam seed_parameters_t the type of seed track parameter
    ///
    /// @param seed_params seed track parameter
    /// @param fitter_state the state of kalman fitter
    template <typename seed_parameters_t>
    TRACCC_HOST_DEVICE void filter(
        const seed_parameters_t& seed_params, state& fitter_state,
        vector_type<intersection_type>&& nav_candidates = {}) {

        // Create propagator
        propagator_type propagator({}, {});

        // Set path limit
        fitter_state.m_aborter_state.set_path_limit(m_cfg.pathlimit);

        // Create propagator state
        typename propagator_type::state propagation(
            seed_params, m_field, m_detector, std::move(nav_candidates));

        // @TODO: Should be removed once detray is fixed to set the
        // volume in the constructor
        propagation._navigation.set_volume(seed_params.surface_link().volume());

        // Set overstep tolerance and stepper constraint
        propagation._stepping().set_overstep_tolerance(
            m_cfg.overstep_tolerance);
        propagation._stepping
            .template set_constraint<detray::step::constraint::e_accuracy>(
                m_cfg.step_constraint);

        // Run forward filtering
        propagator.propagate(propagation, fitter_state());

        // Run smoothing
        smooth(fitter_state);

        // Update track fitting qualities
        update_statistics(fitter_state);
    }

    /// Run smoothing after kalman filtering
    ///
    /// @brief The smoother is based on "Application of Kalman filtering to
    /// track and vertex fitting", R.Frühwirth, NIM A.
    ///
    /// @param fitter_state the state of kalman fitter
    TRACCC_HOST_DEVICE
    void smooth(state& fitter_state) {
        auto& track_states = fitter_state.m_fit_actor_state.m_track_states;

        // The smoothing algorithm requires the following:
        // (1) the filtered track parameter of the current surface
        // (2) the smoothed track parameter of the next surface
        //
        // Since the smoothed track parameter of the last surface can be
        // considered to be the filtered one, we can reversly iterate the
        // algorithm to obtain the smoothed parameter of other surfaces
        auto& last = track_states.back();
        last.smoothed().set_vector(last.filtered().vector());
        last.smoothed().set_covariance(last.filtered().covariance());

        for (typename vector_type<
                 track_state<transform3_type>>::reverse_iterator it =
                 track_states.rbegin() + 1;
             it != track_states.rend(); ++it) {

            // Run kalman smoother
            const detray::surface<detector_type> sf{m_detector,
                                                    it->surface_link()};
            sf.template visit_mask<gain_matrix_smoother<transform3_type>>(
                *it, *(it - 1));
        }
    }

    TRACCC_HOST_DEVICE
    void update_statistics(state& fitter_state) {

        // Reset the ndf and chi2 of fitter info
        fitter_state.m_fit_info.ndf = 0.f;
        fitter_state.m_fit_info.chi2 = 0.f;
        fitter_state.m_fit_info.pval = 0.f;

        auto& fit_info = fitter_state.m_fit_info;
        auto& track_states = fitter_state.m_fit_actor_state.m_track_states;

        // Fit parameter = smoothed track parameter at the first surface
        fit_info.fit_params = track_states[0].smoothed();

        for (const auto& trk_state : track_states) {

            const detray::surface<detector_type> sf{m_detector,
                                                    trk_state.surface_link()};
            sf.template visit_mask<statistics_updater<transform3_type>>(
                fit_info, trk_state);
        }

        // Subtract the NDoF with the degree of freedom of the bound track (=5)
        fit_info.ndf = fit_info.ndf - 5.f;

        // p value
        fit_info.pval = detail::chisquared_cdf_c(fit_info.chi2, fit_info.ndf);
    }

    private:
    // Detector object
    const detector_type& m_detector;
    // Field object
    const bfield_type m_field;

    // Configuration object
    config_type m_cfg;
};

}  // namespace traccc