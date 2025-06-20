/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "propagate_to_next_surface_src.cuh"

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/finding/actors/ckf_aborter.hpp"
#include "traccc/finding/actors/interaction_register.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/detector_type_utils.hpp"
#include "traccc/utils/propagation.hpp"

namespace traccc::cuda {

using interactor_t = detray::pointwise_material_interactor<default_algebra>;
using propagator_t = detray::propagator<
    stepper_for_t<traccc::telescope_detector::device>,
    navigator_for_t<traccc::telescope_detector::device>,
    detray::actor_chain<detray::pathlimit_aborter<scalar>,
                        detray::parameter_transporter<default_algebra>,
                        interaction_register<interactor_t>, interactor_t,
                        detray::momentum_aborter<scalar>, ckf_aborter>>;
using bfield_t = covfie::field<traccc::const_bfield_backend_t<scalar>>::view_t;

template void propagate_to_next_surface<propagator_t, bfield_t>(
    const dim3& grid_size, const dim3& block_size, std::size_t shared_mem_size,
    const cudaStream_t& stream, const finding_config,
    device::propagate_to_next_surface_payload<propagator_t, bfield_t>);

}  // namespace traccc::cuda
