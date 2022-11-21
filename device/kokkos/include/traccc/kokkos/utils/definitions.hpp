/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <Kokkos_Core.hpp>

namespace traccc::kokkos {

// defining MemSpace to generalize in case they use other
#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#endif

#ifndef MemSpace
#define MemSpace Kokkos::HostSpace
#endif

// defining execution space and range_policy
using ExecSpace = MemSpace::execution_space;
using range_policy = Kokkos::RangePolicy<ExecSpace>;

typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
typedef Kokkos::TeamPolicy<ExecSpace>::member_type member_type;

}  // namespace traccc::kokkos