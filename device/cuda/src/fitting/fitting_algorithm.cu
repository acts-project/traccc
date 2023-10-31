/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../utils/utils.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/cuda/utils/definitions.hpp"
#include "traccc/fitting/device/fit.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"

// detray include(s).
#include "detray/core/detector_metadata.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/detectors/telescope_metadata.hpp"
#include "detray/detectors/toy_metadata.hpp"
#include "detray/masks/unbounded.hpp"
#include "detray/propagator/rk_stepper.hpp"

// System include(s).
#include <vector>

namespace traccc::cuda {

namespace kernels {

template <typename fitter_t, typename detector_view_t>
__global__ void fit(
    detector_view_t det_data, const typename fitter_t::bfield_type field_data,
    const typename fitter_t::config_type cfg,
    vecmem::data::jagged_vector_view<typename fitter_t::intersection_type>
        nav_candidates_buffer,
    track_candidate_container_types::const_view track_candidates_view,
    track_state_container_types::view track_states_view) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::fit<fitter_t>(gid, det_data, field_data, cfg, nav_candidates_buffer,
                          track_candidates_view, track_states_view);
}

}  // namespace kernels

template <typename fitter_t>
fitting_algorithm<fitter_t>::fitting_algorithm(
    const config_type& cfg, const traccc::memory_resource& mr,
    vecmem::copy& copy, stream& str)
    : m_cfg(cfg), m_mr(mr), m_copy(copy), m_stream(str){};

template <typename fitter_t>
track_state_container_types::buffer fitting_algorithm<fitter_t>::operator()(
    const typename fitter_t::detector_type::view_type& det_view,
    const typename fitter_t::bfield_type& field_view,
    const vecmem::data::jagged_vector_view<
        typename fitter_t::intersection_type>& navigation_buffer,
    const typename track_candidate_container_types::const_view&
        track_candidates_view) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Number of tracks
    const track_candidate_container_types::const_device::header_vector::
        size_type n_tracks = m_copy.get_size(track_candidates_view.headers);

    // Get the sizes of the track candidates in each track
    const std::vector<track_candidate_container_types::const_device::
                          item_vector::value_type::size_type>
        candidate_sizes = m_copy.get_sizes(track_candidates_view.items);

    track_state_container_types::buffer track_states_buffer{
        {n_tracks, m_mr.main},
        {candidate_sizes, m_mr.main, m_mr.host,
         vecmem::data::buffer_type::resizable}};

    m_copy.setup(track_states_buffer.headers);
    m_copy.setup(track_states_buffer.items);
    m_copy.setup(navigation_buffer);

    // Calculate the number of threads and thread blocks to run the track
    // fitting
    if (n_tracks > 0) {
        const unsigned int nThreads = WARP_SIZE * 2;
        const unsigned int nBlocks = (n_tracks + nThreads - 1) / nThreads;

        // Run the track fitting
        kernels::fit<fitter_t><<<nBlocks, nThreads, 0, stream>>>(
            det_view, field_view, m_cfg, navigation_buffer,
            track_candidates_view, track_states_buffer);
        CUDA_ERROR_CHECK(cudaGetLastError());
    }

    m_stream.synchronize();

    return track_states_buffer;
}

// Explicit template instantiation
using default_detector_type =
    detray::detector<detray::default_metadata, detray::device_container_types>;
using default_stepper_type =
    detray::rk_stepper<covfie::field<detray::bfield::const_bknd_t>::view_t,
                       transform3, detray::constrained_step<>>;
using default_navigator_type = detray::navigator<const default_detector_type>;
using default_fitter_type =
    kalman_fitter<default_stepper_type, default_navigator_type>;
template class fitting_algorithm<default_fitter_type>;

}  // namespace traccc::cuda