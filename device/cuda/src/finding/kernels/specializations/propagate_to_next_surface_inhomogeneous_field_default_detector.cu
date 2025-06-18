/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "propagate_to_next_surface_src.cuh"
#include "types.hpp"

namespace traccc::cuda {

template void propagate_to_next_surface<
    inhomogeneous_field_default_finding_algorithm::propagator_type,
    inhomogeneous_field_default_finding_algorithm::bfield_type>(
    const dim3& grid_size, const dim3& block_size, std::size_t shared_mem_size,
    const cudaStream_t& stream, const finding_config,
    device::propagate_to_next_surface_payload<
        inhomogeneous_field_default_finding_algorithm::propagator_type,
        inhomogeneous_field_default_finding_algorithm::bfield_type>);

}  // namespace traccc::cuda
