/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "full_chain_algorithm.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

// System include(s).
#include <iostream>
#include <stdexcept>

/// Helper macro for checking the return value of CUDA function calls
#define CUDA_ERROR_CHECK(EXP)                                                  \
    do {                                                                       \
        const cudaError_t errorCode = EXP;                                     \
        if (errorCode != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("Failed to run " #EXP " (") + \
                                     cudaGetErrorString(errorCode) + ")");     \
        }                                                                      \
    } while (false)

namespace traccc::cuda {

full_chain_algorithm::full_chain_algorithm(
    vecmem::memory_resource& host_mr,
    const unsigned short target_cells_per_partition)
    : m_host_mr(host_mr),
      m_stream(),
      m_device_mr(),
      m_cached_device_mr(
          std::make_unique<vecmem::binary_page_memory_resource>(m_device_mr)),
      m_copy(m_stream.cudaStream()),
      m_target_cells_per_partition(target_cells_per_partition),
      m_clusterization(memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                       m_stream, m_target_cells_per_partition),
      m_seeding(memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                m_stream),
      m_track_parameter_estimation(
          memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy, m_stream) {

    // Tell the user what device is being used.
    int device = 0;
    CUDA_ERROR_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&props, device));
    std::cout << "Using CUDA device: " << props.name << " [id: " << device
              << ", bus: " << props.pciBusID
              << ", device: " << props.pciDeviceID << "]" << std::endl;
}

full_chain_algorithm::full_chain_algorithm(const full_chain_algorithm& parent)
    : m_host_mr(parent.m_host_mr),
      m_stream(),
      m_device_mr(),
      m_cached_device_mr(
          std::make_unique<vecmem::binary_page_memory_resource>(m_device_mr)),
      m_copy(m_stream.cudaStream()),
      m_target_cells_per_partition(parent.m_target_cells_per_partition),
      m_clusterization(memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                       m_stream, m_target_cells_per_partition),
      m_seeding(memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                m_stream),
      m_track_parameter_estimation(
          memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy, m_stream) {}

full_chain_algorithm::~full_chain_algorithm() {

    // We need to ensure that the caching memory resource would be deleted
    // before the device memory resource that it is based on.
    m_cached_device_mr.reset();
}

full_chain_algorithm::output_type full_chain_algorithm::operator()(
    const cell_collection_types::host& cells,
    const cell_module_collection_types::host& modules) const {

    // Create device copy of input collections
    cell_collection_types::buffer cells_buffer(cells.size(),
                                               *m_cached_device_mr);
    m_copy(vecmem::get_data(cells), cells_buffer);
    cell_module_collection_types::buffer modules_buffer(modules.size(),
                                                        *m_cached_device_mr);
    m_copy(vecmem::get_data(modules), modules_buffer);

    // Run the clusterization (asynchronously).
    const clusterization_algorithm::output_type spacepoints =
        m_clusterization(cells_buffer, modules_buffer);
    const track_params_estimation::output_type track_params =
        m_track_parameter_estimation(spacepoints.first,
                                     m_seeding(spacepoints.first));

    // Get the final data back to the host.
    bound_track_parameters_collection_types::host result;
    m_copy(track_params, result);
    m_stream.synchronize();

    // Return the host container.
    return result;
}

}  // namespace traccc::cuda
