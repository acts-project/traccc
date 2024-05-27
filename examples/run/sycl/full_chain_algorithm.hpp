/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/sycl/clusterization/clusterization_algorithm.hpp"
#include "traccc/sycl/clusterization/spacepoint_formation_algorithm.hpp"
#include "traccc/sycl/seeding/seeding_algorithm.hpp"
#include "traccc/sycl/seeding/track_params_estimation.hpp"
#include "traccc/utils/algorithm.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/memory/sycl/device_memory_resource.hpp>
#include <vecmem/utils/sycl/async_copy.hpp>

// System include(s).
#include <memory>

namespace traccc::sycl {
namespace details {
/// Internal data type used by @c traccc::sycl::full_chain_algorithm
struct full_chain_algorithm_data;
}  // namespace details

/// Algorithm performing the full chain of track reconstruction
///
/// At least as much as is implemented in the project at any given moment.
///
class full_chain_algorithm
    : public algorithm<bound_track_parameters_collection_types::host(
          const cell_collection_types::host&,
          const cell_module_collection_types::host&)> {

    public:
    /// @name (For now dummy...) Type declaration(s)
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

    /// Clustering algorithm type
    using clustering_algorithm = clusterization_algorithm;
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
    ///
    full_chain_algorithm(
        vecmem::memory_resource& host_mr,
        const clustering_config& clustering_config,
        const seedfinder_config& finder_config,
        const spacepoint_grid_config& grid_config,
        const seedfilter_config& filter_config,
        const finding_algorithm::config_type& finding_config = {},
        const fitting_algorithm::config_type& fitting_config = {},
        detector_type* detector = nullptr);

    /// Copy constructor
    ///
    /// An explicit copy constructor is necessary because in the MT tests
    /// we do want to copy such objects, but a default copy-constructor can
    /// not be generated for them.
    ///
    /// @param parent The parent algorithm chain to copy
    ///
    full_chain_algorithm(const full_chain_algorithm& parent);

    /// Algorithm destructor
    ~full_chain_algorithm();

    /// Reconstruct track parameters in the entire detector
    ///
    /// @param cells The cells for every detector module in the event
    /// @return The track parameters reconstructed
    ///
    output_type operator()(
        const cell_collection_types::host& cells,
        const cell_module_collection_types::host& modules) const override;

    private:
    /// Private data object
    details::full_chain_algorithm_data* m_data;
    /// Host memory resource
    vecmem::memory_resource& m_host_mr;
    /// Device memory resource
    std::unique_ptr<vecmem::sycl::device_memory_resource> m_device_mr;
    /// Device caching memory resource
    std::unique_ptr<vecmem::binary_page_memory_resource> m_cached_device_mr;
    /// Memory copy object
    mutable vecmem::sycl::async_copy m_copy;

    /// @name Sub-algorithms used by this full-chain algorithm
    /// @{
    /// Clusterization algorithm
    clusterization_algorithm m_clusterization;
    /// Spacepoint formation algorithm
    spacepoint_formation_algorithm m_spacepoint_formation;
    /// Seeding algorithm
    seeding_algorithm m_seeding;
    /// Track parameter estimation algorithm
    track_params_estimation m_track_parameter_estimation;

    /// Configs
    clustering_config m_clustering_config;
    seedfinder_config m_finder_config;
    spacepoint_grid_config m_grid_config;
    seedfilter_config m_filter_config;

    /// @}

};  // class full_chain_algorithm

}  // namespace traccc::sycl
