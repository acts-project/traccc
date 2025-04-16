/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/utils.hpp"
#include "./kernels/fill_vectors.cuh"
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace traccc::cuda {

greedy_ambiguity_resolution_algorithm::greedy_ambiguity_resolution_algorithm(
    const config_type& cfg, traccc::memory_resource& mr, vecmem::copy& copy,
    stream& str, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_config(cfg),
      m_mr(mr),
      m_copy(copy),
      m_stream(str) {}

greedy_ambiguity_resolution_algorithm::output_type
greedy_ambiguity_resolution_algorithm::operator()(
    const track_candidate_container_types::const_view& track_candidates_view)
    const {

    const track_candidate_container_types::const_device track_candidates(
        track_candidates_view);

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // The Thrust policy to use.
    auto thrust_policy =
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(m_mr.main)))
            .on(stream);

    const track_candidate_container_types::const_view::header_vector::size_type
        n_tracks = m_copy.get().get_size(track_candidates_view.headers);

    if (n_tracks == 0) {
        return track_candidate_container_types::buffer{
            {0, m_mr.main},
            {std::vector<std::size_t>(0, 0), m_mr.main, m_mr.host,
             vecmem::data::buffer_type::resizable}};
    }

    // Make sure that max_shared_meas is largen than zero
    assert(m_config.max_shared_meas > 0u);

    // Status (1 = Accept, 0 = Reject)
    vecmem::data::vector_buffer<int> status_buffer{n_tracks, m_mr.main};
    vecmem::device_vector<int> status_device(status_buffer);
    thrust::fill(thrust_policy, status_device.begin(), status_device.end(), 1);

    // Get the sizes of the track candidates in each track
    using jagged_buffer_size_type = track_candidate_container_types::
        const_device::item_vector::value_type::size_type;
    const std::vector<jagged_buffer_size_type> candidate_sizes =
        m_copy.get().get_sizes(track_candidates_view.items);

    // Make measurement size vector
    std::vector<jagged_buffer_size_type> meas_sizes(n_tracks);
    std::transform(candidate_sizes.begin(), candidate_sizes.end(),
                   meas_sizes.begin(),
                   [this](const jagged_buffer_size_type sz) { return sz; });

    // Make measurement ID, pval and n_measurement vector
    vecmem::data::jagged_vector_buffer<std::size_t> meas_ids_buffer{
        meas_sizes, m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable};

    const auto n_cands_total = track_candidates.total_size();
    vecmem::data::vector_buffer<std::size_t> flat_meas_ids_buffer{
        n_cands_total, m_mr.main, vecmem::data::buffer_type::resizable};
    vecmem::data::vector_buffer<traccc::scalar> pvals_buffer(n_tracks,
                                                             m_mr.main);
    vecmem::data::vector_buffer<std::size_t> n_meas_buffer(n_tracks, m_mr.main);

    {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (n_tracks + nThreads - 1) / nThreads;

        // Fill the vectors
        kernels::fill_vectors<<<nBlocks, nThreads, 0, stream>>>(
            m_config, device::fill_vectors_payload{
                          .track_candidates_view = track_candidates_view,
                          .meas_ids_view = meas_ids_buffer,
                          .flat_meas_ids_view = flat_meas_ids_buffer,
                          .pvals_view = pvals_buffer,
                          .n_meas_view = n_meas_buffer,
                          .status_view = status_buffer});
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        m_stream.get().synchronize();
    }

    // Sort the flat measurement id vector
    thrust::sort(thrust_policy, flat_meas_ids_buffer.ptr(),
                 flat_meas_ids_buffer.ptr() + n_cands_total);

    // Count the number of unique measurements
    const unsigned int meas_count = static_cast<unsigned int>(
        thrust::unique_count(thrust_policy, flat_meas_ids_buffer.ptr(),
                             flat_meas_ids_buffer.ptr() + n_cands_total,
                             thrust::equal_to<int>()));
    vecmem::data::vector_buffer<std::size_t> unique_meas_buffer(meas_count,
                                                                m_mr.main);
    vecmem::data::vector_buffer<std::size_t> unique_meas_counts_buffer(
        meas_count, m_mr.main);

    thrust::reduce_by_key(thrust_policy, flat_meas_ids_buffer.ptr(),
                          flat_meas_ids_buffer.ptr() + n_cands_total,
                          thrust::make_constant_iterator(1),
                          unique_meas_buffer.ptr(),
                          unique_meas_counts_buffer.ptr());

    m_copy.get().setup(unique_meas_counts_buffer)->wait();

    // Copy the unique measurement count to host buffer
    vecmem::data::vector_buffer<std::size_t> unique_meas_counts_host_buffer(
        meas_count, *m_mr.host);
    vecmem::device_vector<std::size_t> unique_meas_counts_device(
        unique_meas_counts_host_buffer);

    m_copy
        .get()(unique_meas_counts_buffer, unique_meas_counts_host_buffer,
               vecmem::copy::type::device_to_host)
        ->wait();

    // Make a host vector
    std::vector<std::size_t> unique_meas_counts;
    unique_meas_counts.reserve(meas_count);
    std::copy(unique_meas_counts_device.begin(),
              unique_meas_counts_device.end(), unique_meas_counts.begin());

    // Fill the tracks per measurement
    vecmem::data::jagged_vector_buffer<std::size_t>
        tracks_per_measurement_buffer(unique_meas_counts, m_mr.main, m_mr.host);

    // Create resolved candidate buffer
    track_candidate_container_types::buffer res_candidates_buffer{
        {10, m_mr.main},
        {std::vector<std::size_t>(10, 10), m_mr.main, m_mr.host,
         vecmem::data::buffer_type::resizable}};

    return res_candidates_buffer;
}

}  // namespace traccc::cuda