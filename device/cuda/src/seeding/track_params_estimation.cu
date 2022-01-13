/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/seeding/track_params_estimation.hpp>
#include <cuda/utils/definitions.hpp>

namespace traccc {
namespace cuda {

/// Forward declaration of track parameter estimating kernel
/// The bound track parameters at the bottom spacepoints are obtained
///
/// @param seeds_view seeds found by seed finding
/// @param params_view vector of bound track parameters at the bottom
/// spacepoints
__global__ void track_params_estimating_kernel(
    spacepoint_container_view spacepoints_view, seed_container_view seeds_view,
    vecmem::data::vector_view<bound_track_parameters> params_view);

track_params_estimation::output_type track_params_estimation::operator()(
    host_spacepoint_container&& spacepoints,
    host_seed_container&& seeds) const {

    auto n_seeds = seeds.get_headers()[0];
    output_type params(n_seeds, &m_mr.get());

    auto spacepoints_view = get_data(spacepoints, &m_mr.get());
    auto seeds_view = get_data(seeds, &m_mr.get());
    auto params_view = vecmem::get_data(params);

    // -- Num threads
    // The dimension of block is the integer multiple of WARP_SIZE (=32)
    unsigned int num_threads = WARP_SIZE * 2;

    // -- Num blocks
    // The dimension of grid is number_of_seeds / num_threads + 1
    unsigned int num_blocks =
        (seeds.get_headers()[0] + num_threads - 1) / num_threads;

    // run the kernel
    track_params_estimating_kernel<<<num_blocks, num_threads>>>(
        spacepoints_view, seeds_view, params_view);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    return params;
}

__global__ void track_params_estimating_kernel(
    spacepoint_container_view spacepoints_view, seed_container_view seeds_view,
    vecmem::data::vector_view<bound_track_parameters> params_view) {

    // Get device container for input parameters
    device_spacepoint_container spacepoints_device(
        {spacepoints_view.headers, spacepoints_view.items});
    device_seed_container seeds_device({seeds_view.headers, seeds_view.items});
    device_bound_track_parameters_collection params_device({params_view});

    // vector index for threads
    unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const auto& n_seeds = seeds_device.get_headers()[0];

    // prevent overflow
    if (gid >= n_seeds) {
        return;
    }

    // convenient assumption on bfield and mass
    // TODO: make use of bfield extension for the future
    vector3 bfield = {0, 0, 2};

    const auto& seed = seeds_device.get_items()[0][gid];
    auto& param = params_device[gid].vector();

    // Get bound track parameter
    param =
        seed_to_bound_vector(spacepoints_device, seed, bfield, PION_MASS_MEV);
}

}  // namespace cuda
}  // namespace traccc
