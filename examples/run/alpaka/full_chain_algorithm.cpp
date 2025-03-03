/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "full_chain_algorithm.hpp"

// System include(s).
#include <iostream>
#include <stdexcept>

namespace traccc::alpaka {

full_chain_algorithm::full_chain_algorithm(
    vecmem::memory_resource& host_mr,
    const clustering_config& clustering_config,
    const seedfinder_config& finder_config,
    const spacepoint_grid_config& grid_config,
    const seedfilter_config& filter_config,
    const finding_algorithm::config_type& finding_config,
    const fitting_algorithm::config_type& fitting_config,
    const silicon_detector_description::host& det_descr,
    host_detector_type* detector, std::unique_ptr<const traccc::Logger> logger)
    : messaging(std::move(logger->clone())),
      m_host_mr(host_mr),
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      m_device_mr(),
#else
      m_device_mr(host_mr),
#endif
      m_copy(),
      m_cached_device_mr(
          std::make_unique<::vecmem::binary_page_memory_resource>(m_device_mr)),
      m_field_vec{0.f, 0.f, finder_config.bFieldInZ},
      m_field(
          detray::bfield::create_const_field<host_detector_type::scalar_type>(
              m_field_vec)),
      m_det_descr(det_descr),
      m_device_det_descr(
          static_cast<silicon_detector_description::buffer::size_type>(
              m_det_descr.get().size()),
          m_device_mr),
      m_detector(detector),
      m_clusterization(memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                       clustering_config),
      m_measurement_sorting(m_copy, logger->cloneWithSuffix("MeasSortingAlg")),
      m_spacepoint_formation(memory_resource{*m_cached_device_mr, &m_host_mr},
                             m_copy, logger->cloneWithSuffix("SpFormationAlg")),
      m_seeding(finder_config, grid_config, filter_config,
                memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                logger->cloneWithSuffix("SeedingAlg")),
      m_track_parameter_estimation(
          memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
          logger->cloneWithSuffix("TrackParamEstAlg")),
      m_finding(finding_config,
                memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                logger->cloneWithSuffix("TrackFindingAlg")),
      m_fitting(fitting_config,
                memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                logger->cloneWithSuffix("TrackFittingAlg")),
      m_clustering_config(clustering_config),
      m_finder_config(finder_config),
      m_grid_config(grid_config),
      m_filter_config(filter_config),
      m_finding_config(finding_config),
      m_fitting_config(fitting_config) {

    traccc::alpaka::get_device_info();

    // Copy the detector (description) to the device.
    m_copy(::vecmem::get_data(m_det_descr.get()), m_device_det_descr)->ignore();
    if (m_detector != nullptr) {
        m_device_detector = detray::get_buffer(detray::get_data(*m_detector),
                                               m_device_mr, m_copy);
        m_device_detector_view = detray::get_data(m_device_detector);
    }
}

full_chain_algorithm::full_chain_algorithm(const full_chain_algorithm& parent)
    : messaging(parent.logger().clone()),
      m_host_mr(parent.m_host_mr),
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      m_device_mr(),
#else
      m_device_mr(parent.m_host_mr),
#endif
      m_copy(),
      m_cached_device_mr(
          std::make_unique<::vecmem::binary_page_memory_resource>(m_device_mr)),
      m_field_vec(parent.m_field_vec),
      m_field(parent.m_field),
      m_det_descr(parent.m_det_descr),
      m_device_det_descr(
          static_cast<silicon_detector_description::buffer::size_type>(
              m_det_descr.get().size()),
          m_device_mr),
      m_detector(parent.m_detector),
      m_clusterization(memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                       parent.m_clustering_config),
      m_measurement_sorting(m_copy,
                            parent.logger().cloneWithSuffix("MeasSortingAlg")),
      m_spacepoint_formation(memory_resource{*m_cached_device_mr, &m_host_mr},
                             m_copy,
                             parent.logger().cloneWithSuffix("SpFormationAlg")),
      m_seeding(parent.m_finder_config, parent.m_grid_config,
                parent.m_filter_config,
                memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                parent.logger().cloneWithSuffix("SeedingAlg")),
      m_track_parameter_estimation(
          memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
          parent.logger().cloneWithSuffix("TrackParamEstAlg")),
      m_finding(parent.m_finding_config,
                memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                parent.logger().cloneWithSuffix("TrackFindingAlg")),
      m_fitting(parent.m_fitting_config,
                memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                parent.logger().cloneWithSuffix("TrackFittingAlg")),
      m_clustering_config(parent.m_clustering_config),
      m_finder_config(parent.m_finder_config),
      m_grid_config(parent.m_grid_config),
      m_filter_config(parent.m_filter_config),
      m_finding_config(parent.m_finding_config),
      m_fitting_config(parent.m_fitting_config) {

    // Copy the detector (description) to the device.
    m_copy(::vecmem::get_data(m_det_descr.get()), m_device_det_descr)->ignore();
    if (m_detector != nullptr) {
        m_device_detector = detray::get_buffer(detray::get_data(*m_detector),
                                               m_device_mr, m_copy);
        m_device_detector_view = detray::get_data(m_device_detector);
    }
}

full_chain_algorithm::~full_chain_algorithm() = default;

full_chain_algorithm::output_type full_chain_algorithm::operator()(
    const edm::silicon_cell_collection::host& cells) const {

    // Create device copy of input collections
    edm::silicon_cell_collection::buffer cells_buffer(
        static_cast<unsigned int>(cells.size()), *m_cached_device_mr);
    m_copy(::vecmem::get_data(cells), cells_buffer)->ignore();

    // Run the clusterization.
    const clusterization_algorithm::output_type measurements =
        m_clusterization(cells_buffer, m_device_det_descr);
    m_measurement_sorting(measurements);

    // If we have a Detray detector, run the track finding and fitting.
    if (m_detector != nullptr) {

        // Run the seed-finding.
        const spacepoint_formation_algorithm::output_type spacepoints =
            m_spacepoint_formation(m_device_detector_view, measurements);
        const track_params_estimation::output_type track_params =
            m_track_parameter_estimation(measurements, spacepoints,
                                         m_seeding(spacepoints), m_field_vec);

        // Run the track finding.
        const finding_algorithm::output_type track_candidates = m_finding(
            m_device_detector_view, m_field, measurements, track_params);

        // Run the track fitting.
        const fitting_algorithm::output_type track_states =
            m_fitting(m_device_detector_view, m_field, track_candidates);

        // Copy a limited amount of result data back to the host.
        output_type result{&m_host_mr};
        m_copy(track_states.headers, result)->wait();
        return result;

    }
    // If not, copy the track parameters back to the host, and return a dummy
    // object.
    else {

        // Copy the measurements back to the host.
        measurement_collection_types::host measurements_host(&m_host_mr);
        m_copy(measurements, measurements_host)->wait();

        // Return an empty object.
        return {};
    }
}

}  // namespace traccc::alpaka
