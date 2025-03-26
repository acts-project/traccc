/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/alpaka/fitting/fitting_algorithm.hpp"

#include "../utils/utils.hpp"
#include "traccc/fitting/device/fill_sort_keys.hpp"
#include "traccc/fitting/device/fit.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/geometry/detector.hpp"

// detray include(s).
#include <detray/detectors/bfield.hpp>
#include <detray/propagator/rk_stepper.hpp>

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// System include(s).
#include <memory_resource>
#include <vector>

namespace traccc::alpaka {

struct FillSortKeysKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        track_candidate_container_types::const_view track_candidates_view,
        vecmem::data::vector_view<device::sort_key> keys_view,
        vecmem::data::vector_view<unsigned int> ids_view) const {

        device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::fill_sort_keys(globalThreadIdx, track_candidates_view,
                               keys_view, ids_view);
    }
};

template <typename fitter_t>
struct fit_payload {

    /**
     * @brief View object to the detector description
     */
    typename fitter_t::detector_type::view_type det_data;

    /**
     * @brief View object to the magnetic field description
     */
    typename fitter_t::bfield_type field_data;

    /**
     * @brief View object to the fitting configuration
     */
    typename fitter_t::config_type cfg;

    /**
     * @brief View object to the input track candidates
     */
    track_candidate_container_types::const_view track_candidates_view;

    /**
     * @brief View object to the input track parameters
     */
    vecmem::data::vector_view<const unsigned int> param_ids_view;

    /**
     * @brief View object to the output track states
     */
    track_state_container_types::view track_states_view;

    /**
     * @brief View object to the output barcode sequence
     */
    vecmem::data::jagged_vector_view<detray::geometry::barcode> barcodes_view;

};

template <typename fitter_t, typename detector_view_t>
struct FitTrackKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const fit_payload<fitter_t> *payload)
        const {

        device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::fit<fitter_t>(globalThreadIdx, payload->det_data,
                              payload->field_data, payload->cfg,
                              payload->track_candidates_view,
                              payload->param_ids_view, payload->track_states_view,
                              payload->barcodes_view);
    }
};

template <typename fitter_t>
fitting_algorithm<fitter_t>::fitting_algorithm(
    const config_type& cfg, const traccc::memory_resource& mr,
    vecmem::copy& copy, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_cfg(cfg), m_mr(mr), m_copy(copy) {}

template <typename fitter_t>
track_state_container_types::buffer fitting_algorithm<fitter_t>::operator()(
    const typename fitter_t::detector_type::view_type& det_view,
    const typename fitter_t::bfield_type& field_view,
    const typename track_candidate_container_types::const_view&
        track_candidates_view) const {

    // Setup alpaka
    auto devHost = ::alpaka::getDevByIdx(::alpaka::Platform<Host>{}, 0u);
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, 0u);
    auto queue = Queue{devAcc};
    Idx threadsPerBlock = getWarpSize<Acc>() * 2;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    auto thrustExecPolicy = thrust::device;
#else
    auto thrustExecPolicy = thrust::host;
#endif

    // Number of tracks
    const track_candidate_container_types::const_device::header_vector::
        size_type n_tracks = m_copy.get_size(track_candidates_view.headers);

    // Get the sizes of the track candidates in each track
    using jagged_buffer_size_type = track_candidate_container_types::
        const_device::item_vector::value_type::size_type;
    const std::vector<jagged_buffer_size_type> candidate_sizes =
        m_copy.get_sizes(track_candidates_view.items);

    track_state_container_types::buffer track_states_buffer{
        {n_tracks, m_mr.main},
        {candidate_sizes, m_mr.main, m_mr.host,
         vecmem::data::buffer_type::resizable}};
    track_state_container_types::view track_states_view(track_states_buffer);

    std::vector<jagged_buffer_size_type> seqs_sizes(candidate_sizes.size());
    std::transform(candidate_sizes.begin(), candidate_sizes.end(),
                   seqs_sizes.begin(),
                   [this](const jagged_buffer_size_type sz) {
                       return std::max(sz * m_cfg.barcode_sequence_size_factor,
                                       m_cfg.min_barcode_sequence_capacity);
                   });
    vecmem::data::jagged_vector_buffer<detray::geometry::barcode> seqs_buffer{
        seqs_sizes, m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable};

    m_copy.setup(track_states_buffer.headers)->ignore();
    m_copy.setup(track_states_buffer.items)->ignore();
    m_copy.setup(seqs_buffer)->ignore();

    // Calculate the number of threads and thread blocks to run the track
    // fitting
    if (n_tracks > 0) {
        const Idx blocksPerGrid =
            (n_tracks + threadsPerBlock - 1) / threadsPerBlock;
        auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        vecmem::data::vector_buffer<device::sort_key> keys_buffer(n_tracks,
                                                                  m_mr.main);
        vecmem::data::vector_buffer<unsigned int> param_ids_buffer(n_tracks,
                                                                   m_mr.main);

        // Get key and value for sorting
        ::alpaka::exec<Acc>(
            queue, workDiv, FillSortKeysKernel{}, track_candidates_view,
            vecmem::get_data(keys_buffer), vecmem::get_data(param_ids_buffer));
        ::alpaka::wait(queue);

        // Sort the key to get the sorted parameter ids
        vecmem::device_vector<device::sort_key> keys_device(keys_buffer);
        vecmem::device_vector<unsigned int> param_ids_device(param_ids_buffer);

        thrust::sort_by_key(thrustExecPolicy, keys_device.begin(),
                            keys_device.end(), param_ids_device.begin());

        // Prepare the payload for the track fitting
        fit_payload<fitter_t> payload{
            det_view, field_view, m_cfg, track_candidates_view,
            vecmem::get_data(param_ids_buffer), track_states_view,
            vecmem::get_data(seqs_buffer)};
        auto bufHost_fitPayload = ::alpaka::allocBuf<fit_payload<fitter_t>, Idx>(
            devHost, 1u);
        fit_payload<fitter_t>* fitPayload =
            ::alpaka::getPtrNative(bufHost_fitPayload);
        *fitPayload = payload;

        auto bufAcc_fitPayload = ::alpaka::allocBuf<fit_payload<fitter_t>, Idx>(
            devAcc, 1u);
        ::alpaka::memcpy(queue, bufAcc_fitPayload, bufHost_fitPayload);
        ::alpaka::wait(queue);

        // Run the track fitting
        ::alpaka::exec<Acc>(
            queue, workDiv,
            FitTrackKernel<fitter_t,
                           typename fitter_t::detector_type::view_type>{},
            ::alpaka::getPtrNative(bufAcc_fitPayload));
        ::alpaka::wait(queue);
    }

    return track_states_buffer;
}

// Explicit template instantiation
using default_detector_type = traccc::default_detector::device;
using default_stepper_type = detray::rk_stepper<
    covfie::field<detray::bfield::const_bknd_t<
        default_detector_type::scalar_type>>::view_t,
    default_detector_type::algebra_type,
    detray::constrained_step<default_detector_type::scalar_type>>;
using default_navigator_type = detray::navigator<const default_detector_type>;
using default_fitter_type =
    kalman_fitter<default_stepper_type, default_navigator_type>;
template class fitting_algorithm<default_fitter_type>;

}  // namespace traccc::alpaka
