/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/sycl/clusterization/clusterization_algorithm.hpp"
#include "traccc/sycl/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/sycl/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/sycl/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/sycl/seeding/seeding_algorithm.hpp"
#include "traccc/sycl/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"
#include "traccc/sycl/seeding/track_params_estimation.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/propagation.hpp"

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
    : public algorithm<vecmem::vector<fitting_result<default_algebra>>(
          const edm::silicon_cell_collection::host&)>,
      public messaging {

    public:
    /// @name (For now dummy...) Type declaration(s)
    /// @{

    /// (Host) Detector type used during track finding and fitting
    using host_detector_type = traccc::default_detector::host;
    /// (Device) Detector type used during track finding and fitting
    using device_detector_type = traccc::default_detector::device;

    /// Spacepoint formation algorithm type
    using spacepoint_formation_algorithm =
        traccc::sycl::silicon_pixel_spacepoint_formation_algorithm;
    /// Clustering algorithm type
    using clustering_algorithm = clusterization_algorithm;
    /// Track finding algorithm type
    using finding_algorithm =
        traccc::sycl::combinatorial_kalman_filter_algorithm;
    /// Track fitting algorithm type
    using fitting_algorithm = traccc::sycl::kalman_fitting_algorithm;

    /// @}

    /// Algorithm constructor
    ///
    /// @param mr The memory resource to use for the intermediate and result
    ///           objects
    ///
    full_chain_algorithm(vecmem::memory_resource& host_mr,
                         const clustering_config& clustering_config,
                         const seedfinder_config& finder_config,
                         const spacepoint_grid_config& grid_config,
                         const seedfilter_config& filter_config,
                         const finding_algorithm::config_type& finding_config,
                         const fitting_algorithm::config_type& fitting_config,
                         const silicon_detector_description::host& det_descr,
                         const magnetic_field& field,
                         host_detector_type* detector,
                         std::unique_ptr<const traccc::Logger> logger);

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
        const edm::silicon_cell_collection::host& cells) const override;

    /// Reconstruct track seeds in the entire detector
    ///
    /// @param cells The cells for every detector module in the event
    /// @return The track seeds reconstructed
    ///
    bound_track_parameters_collection_types::host seeding(
        const edm::silicon_cell_collection::host& cells) const;

    private:
    /// Private data object
    std::unique_ptr<details::full_chain_algorithm_data> m_data;
    /// Host memory resource
    std::reference_wrapper<vecmem::memory_resource> m_host_mr;
    /// Device memory resource
    vecmem::sycl::device_memory_resource m_device_mr;
    /// Device caching memory resource
    mutable vecmem::binary_page_memory_resource m_cached_device_mr;
    /// Memory copy object
    mutable vecmem::sycl::async_copy m_copy;

    /// Constant B field for the (seed) track parameter estimation
    traccc::vector3 m_field_vec;
    /// Constant B field for the track finding and fitting
    magnetic_field m_field;

    /// Detector description
    std::reference_wrapper<const silicon_detector_description::host>
        m_det_descr;
    /// Detector description buffer
    silicon_detector_description::buffer m_device_det_descr;
    /// Host detector
    host_detector_type* m_detector;
    /// Buffer holding the detector's payload on the device
    host_detector_type::buffer_type m_device_detector;
    /// View of the detector's payload on the device
    host_detector_type::view_type m_device_detector_view;

    /// @name Sub-algorithms used by this full-chain algorithm
    /// @{

    /// Clusterization algorithm
    clusterization_algorithm m_clusterization;
    /// Measurement sorting algorithm
    measurement_sorting_algorithm m_measurement_sorting;
    /// Spacepoint formation algorithm
    spacepoint_formation_algorithm m_spacepoint_formation;
    /// Seeding algorithm
    seeding_algorithm m_seeding;
    /// Track parameter estimation algorithm
    track_params_estimation m_track_parameter_estimation;
    /// Track finding algorithm
    finding_algorithm m_finding;
    /// Track fitting algorithm
    fitting_algorithm m_fitting;

    /// @}

    /// @}

    /// @name Algorithm configurations
    /// @{

    /// Configuration for clustering
    clustering_config m_clustering_config;
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

}  // namespace traccc::sycl
