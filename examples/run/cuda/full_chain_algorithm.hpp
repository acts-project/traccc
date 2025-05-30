/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/detector_type_utils.hpp"
#include "traccc/utils/messaging.hpp"
#include "traccc/utils/propagation.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// System include(s).
#include <memory>

namespace traccc::cuda {

/// Algorithm performing the full chain of track reconstruction
///
/// At least as much as is implemented in the project at any given moment.
///
class full_chain_algorithm
    : public algorithm<vecmem::vector<fitting_result<default_algebra>>(
          const edm::silicon_cell_collection::host&)>,
      public messaging {

    public:
    /// @name Type declaration(s)
    /// @{

    /// (Host) Detector type used during track finding and fitting
    using host_detector_type = traccc::default_detector::host;
    /// (Device) Detector type used during track finding and fitting
    using device_detector_type = traccc::default_detector::device;

    using scalar_type = device_detector_type::scalar_type;

    using bfield_type =
        covfie::field<traccc::const_bfield_backend_t<traccc::scalar>>;

    /// Stepper type used by the track finding and fitting algorithms
    using stepper_type =
        detray::rk_stepper<bfield_type::view_t,
                           device_detector_type::algebra_type,
                           detray::constrained_step<scalar_type>>;
    /// Navigator type used by the track finding and fitting algorithms
    using navigator_type = detray::navigator<const device_detector_type>;
    /// Spacepoint formation algorithm type
    using spacepoint_formation_algorithm =
        traccc::cuda::spacepoint_formation_algorithm<
            traccc::default_detector::device>;
    /// Clustering algorithm type
    using clustering_algorithm = traccc::cuda::clusterization_algorithm;
    /// Track finding algorithm type
    using finding_algorithm =
        traccc::cuda::finding_algorithm<stepper_type, navigator_type>;
    /// Track fitting algorithm type
    using fitting_algorithm = traccc::cuda::fitting_algorithm<
        traccc::kalman_fitter<stepper_type, navigator_type>>;

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
                         host_detector_type* detector,
                         std::unique_ptr<const traccc::Logger> logger);

    full_chain_algorithm() = delete;
    full_chain_algorithm(const full_chain_algorithm& other) = delete;
    full_chain_algorithm(full_chain_algorithm&& other) noexcept;
    full_chain_algorithm& operator=(const full_chain_algorithm& other) = delete;
    full_chain_algorithm& operator=(full_chain_algorithm&& other) = delete;

    /// Algorithm destructor
    ~full_chain_algorithm();

    /// Reconstruct track parameters in the entire detector
    ///
    /// @param cells The cells for every detector module in the event
    /// @return The track parameters reconstructed
    ///
    output_type operator()(
        const edm::silicon_cell_collection::host& cells) const override;

    private:
    /// Host memory resource
    vecmem::memory_resource& m_host_mr;
    /// CUDA stream to use
    std::shared_ptr<stream> m_stream;
    /// Device memory resource
    std::shared_ptr<vecmem::cuda::device_memory_resource> m_device_mr;
    /// Device caching memory resource
    std::shared_ptr<vecmem::binary_page_memory_resource> m_cached_device_mr;
    /// (Asynchronous) Memory copy object
    std::shared_ptr<vecmem::cuda::async_copy> m_copy;

    /// Constant B field for the (seed) track parameter estimation
    traccc::vector3 m_field_vec;
    /// Constant B field for the track finding and fitting
    bfield_type m_field;

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
};  // class full_chain_algorithm

}  // namespace traccc::cuda
