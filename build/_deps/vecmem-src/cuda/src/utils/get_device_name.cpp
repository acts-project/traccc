/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "get_device_name.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

// System include(s).
#include <sstream>

namespace vecmem {
namespace cuda {
namespace details {

std::string get_device_name(int device) {

    // Get the device's properties.
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device) != cudaSuccess) {
        return "Unknown";
    }

    // Construct a unique name out of those properties.
    std::ostringstream result;
    result << props.name << " [id: " << device << ", bus: " << props.pciBusID
           << ", device: " << props.pciDeviceID << "]";
    return result.str();
}

}  // namespace details
}  // namespace cuda
}  // namespace vecmem
