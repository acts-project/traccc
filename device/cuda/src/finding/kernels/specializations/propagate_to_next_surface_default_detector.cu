/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "propagate_to_next_surface_src.cuh"

// Project include(s).
#include "traccc/finding/details/combinatorial_kalman_filter_types.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/bfield.hpp"

namespace traccc::cuda {

using bfield_t = covfie::field<traccc::const_bfield_backend_t<scalar>>::view_t;
using propagator_t =
    traccc::details::ckf_propagator_t<default_detector::device, bfield_t>;

template void propagate_to_next_surface<propagator_t, bfield_t>(
    const dim3& grid_size, const dim3& block_size, std::size_t shared_mem_size,
    const cudaStream_t& stream, const finding_config,
    device::propagate_to_next_surface_payload<propagator_t, bfield_t>);

}  // namespace traccc::cuda
