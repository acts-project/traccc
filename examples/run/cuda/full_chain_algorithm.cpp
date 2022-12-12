/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
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

full_chain_algorithm::full_chain_algorithm(vecmem::memory_resource& host_mr)
    : m_host_mr(host_mr),
      m_stream(),
      m_device_mr(),
      m_cached_device_mr(
          std::make_unique<vecmem::binary_page_memory_resource>(m_device_mr)),
      m_copy(m_stream.cudaStream()),
      m_partitioning(m_host_mr),
      m_clusterization(memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                       m_stream),
      m_seeding(memory_resource{*m_cached_device_mr, &m_host_mr}),
      m_track_parameter_estimation(
          memory_resource{*m_cached_device_mr, &m_host_mr}) {

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
      m_partitioning(m_host_mr),
      m_clusterization(memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                       m_stream),
      m_seeding(memory_resource{*m_cached_device_mr, &m_host_mr}),
      m_track_parameter_estimation(
          memory_resource{*m_cached_device_mr, &m_host_mr}) {}

full_chain_algorithm::~full_chain_algorithm() {

    // We need to ensure that the caching memory resource would be deleted
    // before the device memory resource that it is based on.
    m_cached_device_mr.reset();
}

full_chain_algorithm::output_type full_chain_algorithm::operator()(
    const alt_cell_collection_types::host& cells,
    const cell_module_collection_types::host& modules) const {

    // Execute the partitioning algorithm (on the host)
    ccl_partition_collection_types::host partitions =
        m_partitioning(cells, modules);

    // Create device copy of input collections
    alt_cell_collection_types::buffer cells_buffer(cells.size(),
                                                   *m_cached_device_mr);
    m_copy(vecmem::get_data(cells), cells_buffer);
    cell_module_collection_types::buffer modules_buffer(modules.size(),
                                                        *m_cached_device_mr);
    m_copy(vecmem::get_data(modules), modules_buffer);
    ccl_partition_collection_types::buffer partitions_buffer(
        partitions.size(), *m_cached_device_mr);
    m_copy(vecmem::get_data(partitions), partitions_buffer);

    // Synchronize assynchronous copies.
    m_stream.synchronize();

    // Run the clusterization (asynchronously).
    const clusterization_algorithm::output_type spacepoints =
        m_clusterization(cells_buffer, modules_buffer, partitions_buffer);
    const track_params_estimation::output_type track_params =
        m_track_parameter_estimation(spacepoints, m_seeding(spacepoints));

    // Get the final data back to the host.
    bound_track_parameters_collection_types::host result;
    m_copy(track_params, result);
    m_stream.synchronize();

    // Return the host container.
    return result;
}

}  // namespace traccc::cuda
