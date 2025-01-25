/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../utils/calculate1DimNdRange.hpp"
#include "../utils/global_index.hpp"

// Project include(s).
#include "traccc/edm/device/sort_key.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/device/fill_sort_keys.hpp"
#include "traccc/fitting/device/fit.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// oneDPL include(s).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#pragma clang diagnostic ignored "-Wimplicit-int-float-conversion"
#pragma clang diagnostic ignored "-Wsign-compare"
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#pragma clang diagnostic pop

// SYCL include(s).
#include <sycl/sycl.hpp>

namespace traccc::sycl {
namespace kernels {

/// Identifier for the kernel that fills the sorting keys.
struct fill_sort_keys;

}  // namespace kernels

namespace details {

template <typename fitter_t, typename fit_kernel_t>
track_state_container_types::buffer fit_tracks(
    const typename fitter_t::detector_type::view_type& det_view,
    const typename fitter_t::bfield_type& field_view,
    const typename track_candidate_container_types::const_view&
        track_candidates_view,
    const fitting_config& config, const memory_resource& mr, vecmem::copy& copy,
    ::sycl::queue& queue) {

    // Get the number of tracks.
    const track_candidate_container_types::const_device::header_vector::
        size_type n_tracks = copy.get_size(track_candidates_view.headers);

    // Get the number of the track candidates (measurements) in each track.
    const std::vector<track_candidate_container_types::const_device::
                          item_vector::value_type::size_type>
        candidate_sizes = copy.get_sizes(track_candidates_view.items);

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

    // Create the buffers for sorting the parameter IDs.
    vecmem::data::vector_buffer<device::sort_key> keys_buffer(n_tracks,
                                                              mr.main);
    vecmem::data::vector_buffer<unsigned int> param_ids_buffer(n_tracks,
                                                               mr.main);
    vecmem::copy::event_type keys_setup_event = copy.setup(keys_buffer);
    vecmem::copy::event_type param_ids_setup_event =
        copy.setup(param_ids_buffer);
    keys_setup_event->wait();
    param_ids_setup_event->wait();

    // The execution range for the two kernels of the function.
    static constexpr unsigned int localSize = 64;
    ::sycl::nd_range<1> range = calculate1DimNdRange(n_tracks, localSize);

    // Fill the keys and param_ids buffers.
    ::sycl::event fill_keys_event = queue.submit([&](::sycl::handler& h) {
        h.parallel_for<kernels::fill_sort_keys>(
            range,
            [track_candidates_view, keys_view = vecmem::get_data(keys_buffer),
             param_ids_view =
                 vecmem::get_data(param_ids_buffer)](::sycl::nd_item<1> item) {
                device::fill_sort_keys(details::global_index(item),
                                       track_candidates_view, keys_view,
                                       param_ids_view);
            });
    });

    // Sort the key to get the sorted parameter ids
    vecmem::device_vector<device::sort_key> keys_device(keys_buffer);
    vecmem::device_vector<unsigned int> param_ids_device(param_ids_buffer);
    fill_keys_event.wait_and_throw();
    oneapi::dpl::sort_by_key(oneapi::dpl::execution::dpcpp_default,
                             keys_device.begin(), keys_device.end(),
                             param_ids_device.begin());

    // Run the fitting, using the sorted parameter IDs.
    track_state_container_types::view track_states_view = track_states_buffer;
    track_states_headers_setup_event->wait();
    track_states_items_setup_event->wait();
    queue
        .submit([&](::sycl::handler& h) {
            h.parallel_for<fit_kernel_t>(
                range, [det_view, field_view, config, track_candidates_view,
                        param_ids_view = vecmem::get_data(param_ids_buffer),
                        track_states_view](::sycl::nd_item<1> item) {
                    device::fit<fitter_t>(details::global_index(item), det_view,
                                          field_view, config,
                                          track_candidates_view, param_ids_view,
                                          track_states_view);
                });
        })
        .wait_and_throw();

    // Return the fitted tracks.
    return track_states_buffer;
}

}  // namespace details
}  // namespace traccc::sycl
