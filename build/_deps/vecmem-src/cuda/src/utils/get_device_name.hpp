/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <string>

namespace vecmem {
namespace cuda {
namespace details {

/// Get a "fully qualified" name for a given CUDA device
///
/// This function provides a uniform way for printing the names of the
/// devices that various VecMem objects would be interacting with.
///
/// @param device The device ID that the CUDA runtime assigned
/// @return A user friently name for the device
///
std::string get_device_name(int device);

}  // namespace details
}  // namespace cuda
}  // namespace vecmem
