/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation_algorithm.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"
#include "traccc/utils/algorithm.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc {

/// Algorithm performing the full chain of track reconstruction
///
/// At least as much as is implemented in the project at any given moment.
///
class full_chain_algorithm : public algorithm<track_state_container_types::host(
                                 const cell_collection_types::host&,
                                 const cell_module_collection_types::host&)> {

    public:
    /// @name Type declaration(s)
    /// @{

    /// Detector type used during track finding and fitting
    using detector_type = detray::detector<detray::default_metadata,
                                           detray::host_container_types>;

    /// Stepper type used by the track finding and fitting algorithms
    using stepper_type =
        detray::rk_stepper<detray::bfield::const_field_t::view_t,
                           detector_type::algebra_type,
                           detray::constrained_step<>>;
    /// Navigator type used by the track finding and fitting algorithms
    using navigator_type = detray::navigator<const detector_type>;

    using clustering_algorithm = host::clusterization_algorithm;
    /// Track finding algorithm type
    using finding_algorithm =
        traccc::finding_algorithm<stepper_type, navigator_type>;
    /// Track fitting algorithm type
    using fitting_algorithm = traccc::fitting_algorithm<
        traccc::kalman_fitter<stepper_type, navigator_type>>;

    /// @}

    /// Algorithm constructor
    ///
    /// @param mr The memory resource to use for the intermediate and result
    ///           objects
    /// @param dummy This is not used anywhere. Allows templating CPU/Device
    /// algorithm.
    ///
    full_chain_algorithm(vecmem::memory_resource& mr,
                         const clustering_algorithm::config_type& dummy,
                         const seedfinder_config& finder_config,
                         const spacepoint_grid_config& grid_config,
                         const seedfilter_config& filter_config,
                         const finding_algorithm::config_type& finding_config,
                         const fitting_algorithm::config_type& fitting_config,
                         detector_type* detector);

    /// Reconstruct track parameters in the entire detector
    ///
    /// @param cells The cells for every detector module in the event
    /// @return The track parameters reconstructed
    ///
    output_type operator()(
        const cell_collection_types::host& cells,
        const cell_module_collection_types::host& modules) const override;

    private:
    /// Constant B field for the (seed) track parameter estimation
    traccc::vector3 m_field_vec;
    /// Constant B field for the track finding and fitting
    detray::bfield::const_field_t m_field;

    /// Detector
    detector_type* m_detector;

    /// @name Sub-algorithms used by this full-chain algorithm
    /// @{

    /// Clusterization algorithm
    host::clusterization_algorithm m_clusterization;
    /// Spacepoint formation algorithm
    host::spacepoint_formation_algorithm m_spacepoint_formation;
    /// Seeding algorithm
    seeding_algorithm m_seeding;
    /// Track parameter estimation algorithm
    track_params_estimation m_track_parameter_estimation;

    /// Track finding algorithm
    finding_algorithm m_finding;
    /// Track fitting algorithm
    fitting_algorithm m_fitting;

    /// @}

    /// @name Algorithm configurations
    /// @{

    /// Configuration for the seed finding
    seedfinder_config m_finder_config;
    /// Configuration for the spacepoint grid formation
    spacepoint_grid_config m_grid_config;
    /// Configuration for the seed filtering
    seedfilter_config m_filter_config;

    /// Configuration for the track finding
    finding_algorithm::config_type m_finding_config;
    /// Configuration for the track fitting
    fitting_algorithm::config_type m_fitting_config;

    /// @}

};  // class full_chain_algorithm

}  // namespace traccc
