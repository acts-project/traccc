/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "full_chain_algorithm.hpp"

// Alpaka include(s).
#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

// System include(s).
#include <iostream>
#include <stdexcept>

namespace traccc::alpaka {

full_chain_algorithm::full_chain_algorithm(
    vecmem::memory_resource& host_mr,
    const unsigned short target_cells_per_partition,
    const seedfinder_config& finder_config,
    const spacepoint_grid_config& grid_config,
    const seedfilter_config& filter_config)
    : m_host_mr(host_mr),
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      m_device_mr(),
#else
      m_device_mr(host_mr),
#endif
      m_cached_device_mr(
          std::make_unique<vecmem::binary_page_memory_resource>(m_device_mr)),
      m_target_cells_per_partition(target_cells_per_partition),
      m_clusterization(memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                       m_target_cells_per_partition),
      m_measurement_sorting(m_copy),
      m_spacepoint_formation(memory_resource{*m_cached_device_mr, &m_host_mr},
                             m_copy),
      m_seeding(finder_config, grid_config, filter_config,
                memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy),
      m_track_parameter_estimation(
          memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy),
      m_finder_config(finder_config),
      m_grid_config(grid_config),
      m_filter_config(filter_config) {

    // Tell the user what device is being used.
    using Acc = ::alpaka::ExampleDefaultAcc<::alpaka::DimInt<1>, uint32_t>;
    int device = 0;
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, 0u);
    auto const props = ::alpaka::getAccDevProps<Acc>(devAcc);
    std::cout << "Using Alpaka device: " << ::alpaka::getName(devAcc)
              << " [id: " << device << "] " << std::endl;
}

full_chain_algorithm::full_chain_algorithm(const full_chain_algorithm& parent)
    : m_host_mr(parent.m_host_mr),
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      m_device_mr(),
#else
      m_device_mr(parent.m_host_mr),
#endif
      m_copy(),
      m_cached_device_mr(
          std::make_unique<vecmem::binary_page_memory_resource>(m_device_mr)),
      m_target_cells_per_partition(parent.m_target_cells_per_partition),
      m_clusterization(memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                       m_target_cells_per_partition),
      m_measurement_sorting(m_copy),
      m_spacepoint_formation(memory_resource{*m_cached_device_mr, &m_host_mr},
                             m_copy),
      m_seeding(parent.m_finder_config, parent.m_grid_config,
                parent.m_filter_config,
                memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy),
      m_track_parameter_estimation(
          memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy),
      m_finder_config(parent.m_finder_config),
      m_grid_config(parent.m_grid_config),
      m_filter_config(parent.m_filter_config) {
}

full_chain_algorithm::~full_chain_algorithm() = default;

full_chain_algorithm::output_type full_chain_algorithm::operator()(
    const cell_collection_types::host& cells,
    const cell_module_collection_types::host& modules) const {

    // Create device copy of input collections
    cell_collection_types::buffer cells_buffer(cells.size(),
                                               *m_cached_device_mr);
    m_copy(vecmem::get_data(cells), cells_buffer)->ignore();
    cell_module_collection_types::buffer modules_buffer(modules.size(),
                                                        *m_cached_device_mr);
    m_copy(vecmem::get_data(modules), modules_buffer)->ignore();

    // Run the clusterization
    const clusterization_algorithm::output_type measurements =
        m_clusterization(cells_buffer, modules_buffer);
    const spacepoint_formation_algorithm::output_type spacepoints =
        m_spacepoint_formation(m_measurement_sorting(measurements),
                               modules_buffer);
    const track_params_estimation::output_type track_params =
        m_track_parameter_estimation(spacepoints, m_seeding(spacepoints),
                                     {0.f, 0.f, m_finder_config.bFieldInZ});

    // Get the final data back to the host.
    bound_track_parameters_collection_types::host result(&m_host_mr);
    m_copy(track_params, result)->wait();

    // Return the host container.
    return result;
}

}  // namespace traccc::alpaka
