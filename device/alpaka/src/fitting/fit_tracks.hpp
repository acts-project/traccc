/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../utils/utils.hpp"

// Project include(s).
#include "traccc/edm/device/sort_key.hpp"
#include "traccc/edm/track_candidate_container.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/device/fill_sort_keys.hpp"
#include "traccc/fitting/device/fit.hpp"
#include "traccc/fitting/device/fit_backward.hpp"
#include "traccc/fitting/device/fit_forward.hpp"
#include "traccc/fitting/device/fit_prelude.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace traccc::alpaka::details {
namespace kernels {

/// Alpaka kernel functor for @c traccc::device::fill_sort_keys
struct fill_sort_keys {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        edm::track_candidate_collection<default_algebra>::const_view
            track_candidates_view,
        vecmem::data::vector_view<device::sort_key> keys_view,
        vecmem::data::vector_view<unsigned int> ids_view) const {

        const device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::fill_sort_keys(globalThreadIdx, track_candidates_view,
                               keys_view, ids_view);
    }
};

/// Alpaka kernel functor for @c traccc::device::fit_prelude
struct fit_prelude {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        vecmem::data::vector_view<const unsigned int> param_ids_view,
        edm::track_candidate_container<default_algebra>::const_view
            track_candidates_view,
        track_state_container_types::view track_states_view,
        vecmem::data::vector_view<unsigned int> param_liveness_view) const {

        const device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::fit_prelude(globalThreadIdx, param_ids_view,
                            track_candidates_view, track_states_view,
                            param_liveness_view);
    }
};

/// Alpaka kernel functor for @c traccc::device::fit_forward
template <typename fitter_t>
struct fit_forward {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const typename fitter_t::config_type cfg,
        const device::fit_payload<fitter_t>* payload) const {

        const device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::fit_forward<fitter_t>(globalThreadIdx, cfg, *payload);
    }
};

/// Alpaka kernel functor for @c traccc::device::fit_backward
template <typename fitter_t>
struct fit_backward {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const typename fitter_t::config_type cfg,
        const device::fit_payload<fitter_t>* payload) const {

        const device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::fit_backward<fitter_t>(globalThreadIdx, cfg, *payload);
    }
};

}  // namespace kernels

template <typename fitter_t>
track_state_container_types::buffer fit_tracks(
    const typename fitter_t::detector_type::view_type& det_view,
    const typename fitter_t::bfield_type& field_view,
    const edm::track_candidate_container<default_algebra>::const_view&
        track_candidates_view,
    const fitting_config& config, const memory_resource& mr, vecmem::copy& copy,
    Queue& queue) {

    /// Thrust policy to use.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    auto thrustExecPolicy = thrust::device;
#else
    auto thrustExecPolicy = thrust::host;
#endif

    // Number of threads per block to use.
    const Idx threadsPerBlock = getWarpSize<Acc>() * 2;

    // Get the number of tracks.
    const edm::track_candidate_collection<
        default_algebra>::const_device::size_type n_tracks =
        copy.get_size(track_candidates_view.tracks);

    // Get the sizes of the track candidates in each track.
    const std::vector<unsigned int> candidate_sizes =
        copy.get_sizes(track_candidates_view.tracks);

    // Create the result buffer.
    track_state_container_types::buffer track_states_buffer{
        {n_tracks, mr.main},
        {candidate_sizes, mr.main, mr.host,
         vecmem::data::buffer_type::resizable}};
    vecmem::copy::event_type track_states_headers_setup_event =
        copy.setup(track_states_buffer.headers);
    vecmem::copy::event_type track_states_items_setup_event =
        copy.setup(track_states_buffer.items);

    // Return early, if there are no tracks.
    if (n_tracks == 0) {
        track_states_headers_setup_event->wait();
        track_states_items_setup_event->wait();
        return track_states_buffer;
    }

    std::vector<unsigned int> seqs_sizes(candidate_sizes.size());
    std::transform(candidate_sizes.begin(), candidate_sizes.end(),
                   seqs_sizes.begin(), [&config](const unsigned int sz) {
                       return std::max(sz * config.barcode_sequence_size_factor,
                                       config.min_barcode_sequence_capacity);
                   });
    vecmem::data::jagged_vector_buffer<detray::geometry::barcode> seqs_buffer{
        seqs_sizes, mr.main, mr.host, vecmem::data::buffer_type::resizable};
    copy.setup(seqs_buffer)->wait();

    // Create the buffers for sorting the parameter IDs.
    vecmem::data::vector_buffer<device::sort_key> keys_buffer(n_tracks,
                                                              mr.main);
    vecmem::data::vector_buffer<unsigned int> param_ids_buffer(n_tracks,
                                                               mr.main);
    vecmem::data::vector_buffer<unsigned int> param_liveness_buffer(n_tracks,
                                                                    mr.main);
    vecmem::copy::event_type keys_setup_event = copy.setup(keys_buffer);
    vecmem::copy::event_type param_ids_setup_event =
        copy.setup(param_ids_buffer);
    vecmem::copy::event_type param_liveness_setup_event =
        copy.setup(param_liveness_buffer);
    keys_setup_event->wait();
    param_ids_setup_event->wait();
    param_liveness_setup_event->wait();

    // The execution range for the two kernels of the function.
    const Idx blocksPerGrid =
        (n_tracks + threadsPerBlock - 1) / threadsPerBlock;
    const auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

    // Fill the keys and param_ids buffers.
    ::alpaka::exec<Acc>(
        queue, workDiv, kernels::fill_sort_keys{}, track_candidates_view.tracks,
        vecmem::get_data(keys_buffer), vecmem::get_data(param_ids_buffer));
    ::alpaka::wait(queue);

    // Sort the key to get the sorted parameter ids
    vecmem::device_vector<device::sort_key> keys_device(keys_buffer);
    vecmem::device_vector<unsigned int> param_ids_device(param_ids_buffer);
    thrust::sort_by_key(thrustExecPolicy, keys_device.begin(),
                        keys_device.end(), param_ids_device.begin());

    // Run the fitting, using the sorted parameter IDs.
    track_state_container_types::view track_states_view = track_states_buffer;
    track_states_headers_setup_event->wait();
    track_states_items_setup_event->wait();

    ::alpaka::exec<Acc>(queue, workDiv, kernels::fit_prelude{},
                        vecmem::get_data(param_ids_buffer),
                        track_candidates_view, track_states_view,
                        vecmem::get_data(param_liveness_buffer));
    ::alpaka::wait(queue);

    // Allocate the fitting kernels's payload in host memory.
    device::fit_payload<fitter_t> host_payload{
        .det_data = det_view,
        .field_data = field_view,
        .param_ids_view = param_ids_buffer,
        .param_liveness_view = param_liveness_buffer,
        .track_states_view = track_states_view,
        .barcodes_view = seqs_buffer};
    // Now copy it to device memory.
    vecmem::data::vector_buffer<device::fit_payload<fitter_t>> device_payload(
        1u, mr.main);
    copy.setup(device_payload)->wait();
    copy(vecmem::data::vector_view<const device::fit_payload<fitter_t>>(
             1u, &host_payload),
         device_payload)
        ->wait();

    for (std::size_t i = 0; i < config.n_iterations; ++i) {
        // Run the track fitting
        ::alpaka::exec<Acc>(queue, workDiv, kernels::fit_forward<fitter_t>{},
                            config, device_payload.ptr());
        ::alpaka::wait(queue);
        ::alpaka::exec<Acc>(queue, workDiv, kernels::fit_backward<fitter_t>{},
                            config, device_payload.ptr());
        ::alpaka::wait(queue);
    }

    // Return the fitted tracks.
    return track_states_buffer;
}

}  // namespace traccc::alpaka::details
