/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/definitions.hpp"

namespace traccc {
namespace cuda {

/// Forward declaration of track parameter estimating kernel
/// The bound track parameters at the bottom spacepoints are obtained
///
/// @param seeds_view seeds found by seed finding
/// @param params_view vector of bound track parameters at the bottom
/// spacepoints
__global__ void track_params_estimating_kernel(
    spacepoint_container_types::const_view spacepoints_view,
    vecmem::data::vector_view<seed> seeds_view,
    vecmem::data::vector_view<bound_track_parameters> params_view);

track_params_estimation::output_type track_params_estimation::operator()(
    const spacepoint_container_types::host& spacepoints,
    host_seed_collection&& seeds) const {

    output_type params(seeds.size(), &m_mr.get());

    spacepoint_container_types::const_data spacepoints_view =
        get_data(spacepoints, &m_mr.get());
    auto seeds_view = vecmem::get_data(seeds);
    auto params_view = vecmem::get_data(params);

    // -- Num threads
    // The dimension of block is the integer multiple of WARP_SIZE (=32)
    unsigned int num_threads = WARP_SIZE * 2;

    // -- Num blocks
    // The dimension of grid is number_of_seeds / num_threads + 1
    unsigned int num_blocks = seeds.size() / num_threads + 1;

    // run the kernel
    track_params_estimating_kernel<<<num_blocks, num_threads>>>(
        spacepoints_view, seeds_view, params_view);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    return params;
}

__global__ void track_params_estimating_kernel(
    spacepoint_container_types::const_view spacepoints_view,
    vecmem::data::vector_view<seed> seeds_view,
    vecmem::data::vector_view<bound_track_parameters> params_view) {

    // Get device container for input parameters
    const spacepoint_container_types::const_device spacepoints_device(
        spacepoints_view);
    device_seed_collection seeds_device(seeds_view);
    device_bound_track_parameters_collection params_device(params_view);

    // vector index for threads
    unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x;

    // prevent overflow
    if (gid >= seeds_device.size()) {
        return;
    }

    // convenient assumption on bfield and mass
    // TODO: make use of bfield extension for the future
    vector3 bfield = {0, 0, 2};

    const auto& seed = seeds_device.at(gid);
    auto& param = params_device[gid].vector();

    // Get bound track parameter
    param =
        seed_to_bound_vector(spacepoints_device, seed, bfield, PION_MASS_MEV);
}

}  // namespace cuda
}  // namespace traccc
