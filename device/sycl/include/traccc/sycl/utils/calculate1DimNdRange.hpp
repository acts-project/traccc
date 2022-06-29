/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Sycl include(s).
#include <CL/sycl.hpp>

namespace traccc::sycl {

/// Function that calculates 1 dim nd range for sycl kernel execution
///
/// @param[in] globalSize       The global execution range of the kernel
/// @param[in] localSize        Desired Work group size of the kernel
/// @return                     The 1 dim sycl nd_range object
///
::sycl::nd_range<1> calculate1DimNdRange(const std::size_t globalSize,
                                         const std::size_t localSize);

}  // namespace traccc::sycl