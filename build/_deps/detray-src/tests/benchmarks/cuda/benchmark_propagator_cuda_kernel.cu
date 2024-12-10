/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "benchmark_propagator_cuda_kernel.hpp"
#include "detray/definitions/detail/cuda_definitions.hpp"

namespace detray {

__global__ void __launch_bounds__(256, 4) propagator_benchmark_kernel(
    typename detector_host_type::view_type det_data,
    covfie::field_view<bfield::const_bknd_t> field_data,
    vecmem::data::vector_view<free_track_parameters<algebra_t>> tracks_data,
    const propagate_option opt) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    detector_device_type det(det_data);
    vecmem::device_vector<free_track_parameters<algebra_t>> tracks(tracks_data);

    if (gid >= tracks.size()) {
        return;
    }

    // Create propagator
    propagation::config cfg{};
    cfg.navigation.search_window = {3u, 3u};
    propagator_device_type p{cfg};

    parameter_transporter<algebra_t>::state transporter_state{};
    pointwise_material_interactor<algebra_t>::state interactor_state{};
    parameter_resetter<algebra_t>::state resetter_state{};

    // Create the actor states
    auto actor_states =
        detray::tie(transporter_state, interactor_state, resetter_state);
    // Create the propagator state
    propagator_device_type::state p_state(tracks.at(gid), field_data, det);

    // Run propagation
    if (opt == propagate_option::e_unsync) {
        p.propagate(p_state, actor_states);
    } else if (opt == propagate_option::e_sync) {
        p.propagate_sync(p_state, actor_states);
    }
}

void propagator_benchmark(
    typename detector_host_type::view_type det_data,
    covfie::field_view<bfield::const_bknd_t> field_data,
    vecmem::data::vector_view<free_track_parameters<algebra_t>>& tracks_data,
    const propagate_option opt) {

    constexpr int thread_dim = 256;
    int block_dim =
        static_cast<int>(tracks_data.size() + thread_dim - 1) / thread_dim;

    // run the test kernel
    propagator_benchmark_kernel<<<block_dim, thread_dim>>>(det_data, field_data,
                                                           tracks_data, opt);

    // cuda error check
    DETRAY_CUDA_ERROR_CHECK(cudaGetLastError());
    DETRAY_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

}  // namespace detray
