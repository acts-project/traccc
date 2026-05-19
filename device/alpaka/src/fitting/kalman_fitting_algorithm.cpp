/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/fitting/kalman_fitting_algorithm.hpp"

#include "../utils/get_queue.hpp"
#include "../utils/magnetic_field_types.hpp"
#include "../utils/parallel_algorithms.hpp"
#include "../utils/utils.hpp"

// Project include(s).
#include "traccc/fitting/details/kalman_fitting_types.hpp"
#include "traccc/fitting/device/fill_fitting_sort_keys.hpp"
#include "traccc/fitting/device/fit.hpp"
#include "traccc/fitting/device/fit_backward.hpp"
#include "traccc/fitting/device/fit_forward.hpp"
#include "traccc/fitting/device/fit_prelude.hpp"
#include "traccc/utils/detector_buffer_bfield_visitor.hpp"

namespace traccc::alpaka {
namespace kernels {

/// Alpaka kernel functor for @c traccc::device::fill_fitting_sort_keys
struct fill_fitting_sort_keys {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        edm::track_collection<default_algebra>::const_view
            track_candidates_view,
        vecmem::data::vector_view<device::sort_key> keys_view,
        vecmem::data::vector_view<unsigned int> ids_view) const {

        const device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::fill_fitting_sort_keys(globalThreadIdx, track_candidates_view,
                                       keys_view, ids_view);
    }
};

/// Alpaka kernel functor for @c traccc::device::fit_prelude
struct fit_prelude {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        vecmem::data::vector_view<const unsigned int> param_ids_view,
        edm::track_container<default_algebra>::const_view track_candidates_view,
        edm::track_container<default_algebra>::view track_states_view,
        vecmem::data::vector_view<unsigned int> param_liveness_view) const {

        const device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::fit_prelude<default_algebra>(
            globalThreadIdx, param_ids_view, track_candidates_view,
            track_states_view, param_liveness_view);
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

kalman_fitting_algorithm::kalman_fitting_algorithm(
    const config_type& config, const traccc::memory_resource& mr,
    const vecmem::copy& copy, alpaka::queue& q,
    std::unique_ptr<const Logger> logger)
    : device::kalman_fitting_algorithm{config, mr, copy, std::move(logger)},
      alpaka::algorithm_base{q} {}

void kalman_fitting_algorithm::prepare_track_fit_order(
    const edm::track_collection<default_algebra>::const_view& tracks,
    vecmem::data::vector_view<device::sort_key>& track_sort_keys,
    vecmem::data::vector_view<unsigned int>& track_indices) const {

    // Get the number of tracks.
    const unsigned int n_tracks = tracks.capacity();
    assert(n_tracks == copy().get_size(tracks));
    assert(n_tracks == track_indices.capacity());
    assert(track_indices.size_ptr() == nullptr);

    // Launch parameters for the kernel.
    const unsigned int nThreads = warp_size() * 4;
    const unsigned int nBlocks = (n_tracks + nThreads - 1) / nThreads;
    auto workDiv = makeWorkDiv<Acc>(nBlocks, nThreads);

    // Fill the keys and indices buffers.
    ::alpaka::exec<Acc>(details::get_queue(queue()), workDiv,
                        kernels::fill_fitting_sort_keys{}, tracks,
                        track_sort_keys, track_indices);

    // Sort the key to get the sorted parameter ids
    vecmem::device_vector<device::sort_key> keys_device(track_sort_keys);
    vecmem::device_vector<unsigned int> track_indices_device(track_indices);
    details::sort_by_key(details::get_queue(queue()), mr(), keys_device.begin(),
                         keys_device.end(), track_indices_device.begin());
}

void kalman_fitting_algorithm::fit_prelude_kernel(
    const vecmem::data::vector_view<const unsigned int>& track_indices,
    const edm::track_container<default_algebra>::const_view& input_tracks,
    edm::track_container<default_algebra>::view output_tracks,
    vecmem::data::vector_view<unsigned int>& track_liveness) const {

    // Get the number of tracks.
    const unsigned int n_tracks = input_tracks.tracks.capacity();
    assert(n_tracks == copy().get_size(input_tracks.tracks));
    assert(n_tracks == track_indices.capacity());
    assert(track_indices.size_ptr() == nullptr);
    assert(n_tracks == copy().get_size(output_tracks.tracks));

    // Launch parameters for the kernel.
    const unsigned int nThreads = warp_size() * 4;
    const unsigned int nBlocks = (n_tracks + nThreads - 1) / nThreads;
    auto workDiv = makeWorkDiv<Acc>(nBlocks, nThreads);

    // Run the fitting, using the sorted parameter IDs.
    ::alpaka::exec<Acc>(details::get_queue(queue()), workDiv,
                        kernels::fit_prelude{}, track_indices, input_tracks,
                        output_tracks, track_liveness);
}

auto kalman_fitting_algorithm::prepare_fit_payload(
    const detector_buffer& det, const magnetic_field& field,
    const std::vector<unsigned int>& n_surfaces,
    const vecmem::data::vector_view<const unsigned int>& track_indices,
    vecmem::data::vector_view<unsigned int>& track_liveness,
    edm::track_container<default_algebra>::view tracks) const
    -> std::unique_ptr<fit_payload_base> {

    return prepare_fit_payload_helper<detector_type_list,
                                      alpaka::bfield_type_list<scalar>>(
        det, field, n_surfaces, track_indices, track_liveness, tracks);
}

void kalman_fitting_algorithm::fit_forward_kernel(
    const fitting_config& config, const fit_payload_base& payload) const {

    return detector_buffer_magnetic_field_visitor<
        detector_type_list, alpaka::bfield_type_list<scalar>>(
        payload.detector, payload.field,
        [&]<typename detector_traits_t, typename bfield_view_t>(
            const typename detector_traits_t::view&, const bfield_view_t&) {
            // Cast the payload to the correct type.
            const fit_payload<typename detector_traits_t::device,
                              bfield_view_t>& fit_payload =
                cast_fit_payload<typename detector_traits_t::device,
                                 bfield_view_t>(payload);

            // Get the number of tracks.
            const unsigned int n_tracks =
                fit_payload.host_payload.tracks_view.tracks.capacity();
            assert(
                n_tracks ==
                copy().get_size(fit_payload.host_payload.tracks_view.tracks));

            // Launch parameters for the kernel.
            const unsigned int nThreads = warp_size() * 4;
            const unsigned int nBlocks = (n_tracks + nThreads - 1) / nThreads;
            auto workDiv = makeWorkDiv<Acc>(nBlocks, nThreads);

            // Fitter type to use.
            using fitter_t = traccc::details::kalman_fitter_t<
                typename detector_traits_t::device, bfield_view_t>;

            // Run the track fitting
            ::alpaka::exec<Acc>(details::get_queue(queue()), workDiv,
                                kernels::fit_forward<fitter_t>{}, config,
                                fit_payload.device_payload.ptr());
        });
}

void kalman_fitting_algorithm::fit_backward_kernel(
    const fitting_config& config, const fit_payload_base& payload) const {

    return detector_buffer_magnetic_field_visitor<
        detector_type_list, alpaka::bfield_type_list<scalar>>(
        payload.detector, payload.field,
        [&]<typename detector_traits_t, typename bfield_view_t>(
            const typename detector_traits_t::view&, const bfield_view_t&) {
            // Cast the payload to the correct type.
            const fit_payload<typename detector_traits_t::device,
                              bfield_view_t>& fit_payload =
                cast_fit_payload<typename detector_traits_t::device,
                                 bfield_view_t>(payload);

            // Get the number of tracks.
            const unsigned int n_tracks =
                fit_payload.host_payload.tracks_view.tracks.capacity();
            assert(
                n_tracks ==
                copy().get_size(fit_payload.host_payload.tracks_view.tracks));

            // Launch parameters for the kernel.
            const unsigned int nThreads = warp_size() * 4;
            const unsigned int nBlocks = (n_tracks + nThreads - 1) / nThreads;
            auto workDiv = makeWorkDiv<Acc>(nBlocks, nThreads);

            // Fitter type to use.
            using fitter_t = traccc::details::kalman_fitter_t<
                typename detector_traits_t::device, bfield_view_t>;

            // Run the track fitting
            ::alpaka::exec<Acc>(details::get_queue(queue()), workDiv,
                                kernels::fit_backward<fitter_t>{}, config,
                                fit_payload.device_payload.ptr());
        });
}

}  // namespace traccc::alpaka
