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
    : m_device_mr(),
      m_copy(),
      m_host2device(memory_resource{m_device_mr, &host_mr}, m_copy),
      m_clusterization(memory_resource{m_device_mr, &host_mr}),
      m_seeding(memory_resource{m_device_mr, &host_mr}),
      m_track_parameter_estimation(memory_resource{m_device_mr, &host_mr}) {

    // Tell the user what device is being used.
    int device = 0;
    CUDA_ERROR_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&props, device));
    std::cout << "Using CUDA device: " << props.name << " [id: " << device
              << ", bus: " << props.pciBusID
              << ", device: " << props.pciDeviceID << "]" << std::endl;
}

full_chain_algorithm::output_type full_chain_algorithm::operator()(
    const cell_container_types::host& cells) const {

    // Execute the algorithms.
    const clusterization_algorithm::output_type spacepoints =
        m_clusterization(m_host2device(get_data(cells)));
    const track_params_estimation::output_type track_params =
        m_track_parameter_estimation(spacepoints, m_seeding(spacepoints));

    // Get the final data back to the host.
    bound_track_parameters_collection_types::host result;
    m_copy(track_params, result);

    // Return the host container.
    return result;
}

}  // namespace traccc::cuda
