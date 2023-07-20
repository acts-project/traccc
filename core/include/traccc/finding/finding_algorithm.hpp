/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/finding/interaction_register.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// detray include(s).
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/propagator.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// Thrust Library
#include <thrust/pair.h>

namespace traccc {

/// Track Finding algorithm for a set of tracks
template <typename stepper_t, typename navigator_t>
class finding_algorithm
    : public algorithm<track_candidate_container_types::host(
          const typename navigator_t::detector_type&,
          const measurement_container_types::host&,
          const bound_track_parameters_collection_types::host&)> {

    /// Transform3 type
    using transform3_type = typename stepper_t::transform3_type;

    /// Detector type
    using detector_type = typename navigator_t::detector_type;

    /// Actor types
    using transporter = detray::parameter_transporter<transform3_type>;
    using interactor = detray::pointwise_material_interactor<transform3_type>;

    /// scalar type
    using scalar_type = typename transform3_type::scalar_type;

    /// Actor chain for propagate to the next surface and its propagator type
    using actor_type =
        detray::actor_chain<std::tuple, detray::pathlimit_aborter, transporter,
                            interaction_register<interactor>, interactor,
                            detray::next_surface_aborter>;

    using propagator_type =
        detray::propagator<stepper_t, navigator_t, actor_type>;

    using intersection_type =
        detray::intersection2D<typename detector_type::surface_type,
                               transform3_type>;

    using interactor_type =
        detray::pointwise_material_interactor<transform3_type>;

    public:
    /// Configuration type
    using config_type = finding_config<scalar_type>;

    /// Constructor for the finding algorithm
    ///
    /// @param cfg  Configuration object
    /// @param mr   The memory resource to use
    finding_algorithm(const config_type& cfg) : m_cfg(cfg) {}

    /// Get config object (const access)
    const finding_config<scalar_type>& get_config() const { return m_cfg; }

    /// Run the algorithm
    ///
    /// @param det    Detector
    /// @param measurements  Input measurements
    /// @param seeds  Input seeds
    track_candidate_container_types::host operator()(
        const detector_type& det,
        const measurement_container_types::host& measurements,
        const bound_track_parameters_collection_types::host& seeds) const;

    private:
    /// Config object
    config_type m_cfg;
};

}  // namespace traccc

#include "traccc/finding/finding_algorithm.ipp"
