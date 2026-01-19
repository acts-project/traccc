/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../sanity/contiguous_on.cuh"
#include "../utils/barrier.hpp"
#include "../utils/cuda_error_handling.hpp"
#include "../utils/get_size.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"
#include "./kernels/kalman_track_follower.hpp"

// Project include(s).
#include "traccc/edm/device/identity_projector.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/details/kalman_track_follower_types.hpp"
#include "traccc/finding/device/barcode_surface_comparator.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/finding/track_state_candidate.hpp"
#include "traccc/utils/logging.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/projections.hpp"
#include "traccc/utils/propagation.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace traccc::cuda::details {

/// Templated implementation of a Kalman Filter based track following algorithm.
///
/// Concrete track finding algorithms can use this function with the appropriate
/// specializations, to find tracks on top of a specific detector type, magnetic
/// field type, and track finding configuration.
///
/// @tparam detector_t The (device) detector type to use
/// @tparam bfield_t   The magnetic field type to use
///
/// @param det               A view of the detector object
/// @param field             The magnetic field object
/// @param measurements_view All measurements in an event
/// @param seeds_view        All seeds in an event to start the track finding
///                          with
/// @param config            The track finding configuration
/// @param mr                The memory resource(s) to use
/// @param copy              The copy object to use
/// @param log               The logger to use for message logging
/// @param stream            The CUDA stream to use for the operations
/// @param warp_size         The warp size of the used CUDA device
///
/// @return A buffer of the found track candidates
///
template <typename detector_t, typename bfield_t>
edm::track_container<typename detector_t::algebra_type>::buffer
kalman_track_follower(
    const typename detector_t::const_view_type& det, const bfield_t& field,
    const typename edm::measurement_collection<
        typename detector_t::algebra_type>::const_view& measurements_view,
    const bound_track_parameters_collection_types::const_view& seeds,
    const finding_config& config, const memory_resource& mr, vecmem::copy& copy,
    const Logger& /*log*/, stream& str, unsigned int warp_size) {

    using algebra_t = typename detector_t::algebra_type;
    using scalar_t = detray::dscalar<algebra_t>;

    using propagator_t = traccc::details::kf_propagator_t<detector_t, bfield_t>;

    // Create a logger.
    // auto logger = [&log]() -> const Logger& { return log; };

    /// Access the underlying CUDA stream.
    cudaStream_t stream = get_stream(str);

    /// Thrust policy to use.
    auto thrust_policy =
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr.main)))
            .on(stream);

    /*****************************************************************
     * Measurement Operations
     *****************************************************************/

    const typename edm::measurement_collection<algebra_t>::const_device
        measurements{measurements_view};

    assert(is_contiguous_on<
           vecmem::device_vector<const detray::geometry::barcode>>(
        device::identity_projector{}, mr.main, copy, str,
        measurements_view.template get<6>()));

    const auto n_measurements = copy.get_size(measurements_view);

    // Access the detector view as a detector object
    detector_t device_det(det);
    const unsigned int n_surfaces{device_det.surfaces().size()};

    // Get upper bounds of measurement ranges per surface
    vecmem::data::vector_buffer<unsigned int> meas_ranges_buffer{n_surfaces,
                                                                 mr.main};
    copy.setup(meas_ranges_buffer)->ignore();
    vecmem::device_vector<unsigned int> measurement_ranges(meas_ranges_buffer);

    thrust::upper_bound(thrust_policy, measurements.surface_link().begin(),
                        // We have to use this ugly form here, because if the
                        // measurement collection is resizable (which it often
                        // is), the end() function cannot be used in host code.
                        measurements.surface_link().begin() + n_measurements,
                        device_det.surfaces().begin(),
                        device_det.surfaces().end(), measurement_ranges.begin(),
                        device::barcode_surface_comparator{});

    /*****************************************************************
     * Run propagation
     *****************************************************************/

    // Prepare input parameters with seeds
    const unsigned int n_seeds = copy.get_size(seeds);

    bound_track_parameters_collection_types::buffer seeds_buffer(n_seeds,
                                                                 mr.main);
    copy.setup(seeds_buffer)->ignore();
    copy(seeds, seeds_buffer, vecmem::copy::type::device_to_device)->ignore();

    // Get output track statistics
    vecmem::data::vector_buffer<track_stats<scalar_t>> track_stats_buffer{
        n_seeds, mr.main};
    copy.setup(track_stats_buffer)->ignore();

    // Get the output track state data, depending on which data is required
    vecmem::data::vector_buffer<track_state_candidate> track_cand_buffer{
        0, mr.main};
    vecmem::data::vector_buffer<filtered_track_state_candidate<algebra_t>>
        filtered_track_cand_buffer{0, mr.main};
    vecmem::data::vector_buffer<full_track_state_candidate<algebra_t>>
        full_track_cand_buffer{0, mr.main};

    // Allocate memory for the required data collection mode
    const unsigned int max_cands{seeds.size() *
                                 config.max_track_candidates_per_track};
    if (config.run_smoother == smoother_type::e_none) {
        track_cand_buffer = vecmem::data::vector_buffer<track_state_candidate>{
            max_cands, mr.main};
    } else if (config.run_smoother == smoother_type::e_kalman) {
        filtered_track_cand_buffer = vecmem::data::vector_buffer<
            filtered_track_state_candidate<algebra_t>>{max_cands, mr.main};
    } else if (config.run_smoother == smoother_type::e_mbf) {
        full_track_cand_buffer =
            vecmem::data::vector_buffer<full_track_state_candidate<algebra_t>>{
                max_cands, mr.main};
    }

    // Allocate the kernel's payload in host memory.
    using payload_t = device::kalman_track_follower_payload<propagator_t>;
    const payload_t host_payload{
        .det_data = det,
        .field_data = field,
        .seeds_view = seeds_buffer,
        .measurements_view = measurements_view,
        .measurement_ranges_view = meas_ranges_buffer,
        .track_stats_view = track_stats_buffer,
        .track_cand_view = track_cand_buffer,
        .filtered_track_cand_view = filtered_track_cand_buffer,
        .full_track_cand_view = full_track_cand_buffer,
    };

    const unsigned int nThreads = warp_size * 4;
    const unsigned int nBlocks = (n_seeds - 1) / nThreads;
    traccc::cuda::kalman_track_follower<propagator_t>(
        nBlocks, nThreads, 0u, stream, config, host_payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    str.synchronize();

    // Create track candidate buffer
    vecmem::vector<unsigned int> n_constituent_links(mr.host);
    n_constituent_links.resize(n_seeds, config.max_track_candidates_per_track);

    typename edm::track_container<algebra_t>::buffer track_candidates_buffer{
        {n_constituent_links, mr.main, mr.host},
        {n_seeds * config.max_track_candidates_per_track, mr.main,
         vecmem::data::buffer_type::resizable},
        measurements_view};
    copy.setup(track_candidates_buffer.tracks)->ignore();
    copy.setup(track_candidates_buffer.states)->ignore();

    return track_candidates_buffer;
}

}  // namespace traccc::cuda::details
