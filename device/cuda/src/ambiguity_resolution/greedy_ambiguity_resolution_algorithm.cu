/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/utils.hpp"
#include "./kernels/count_shared_measurements.cuh"
#include "./kernels/fill_tracks_per_measurement.cuh"
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
      m_stream(str),
      m_warp_size(details::get_warp_size(str.device())) {}

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

    printf("n tracks: %d \n", n_tracks);

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

    printf("status size %d \n", m_copy.get().get_size(status_buffer));
    vecmem::device_vector<int> status_device(status_buffer);
    thrust::fill(thrust_policy, status_device.begin(), status_device.end(), 1);

    m_stream.get().synchronize();

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

    for (const auto& e : meas_sizes) {
        printf("meas size %d \n", e);
    }

    // Make measurement ID, pval and n_measurement vector
    vecmem::data::jagged_vector_buffer<std::size_t> meas_ids_buffer{
        meas_sizes, m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable};
    m_copy.get().setup(meas_ids_buffer)->ignore();

    const auto n_cands_total = track_candidates.total_size();

    printf("n cands total: %d \n", n_cands_total);

    vecmem::data::vector_buffer<std::size_t> flat_meas_ids_buffer{
        n_cands_total, m_mr.main, vecmem::data::buffer_type::resizable};
    vecmem::data::vector_buffer<traccc::scalar> pvals_buffer{n_tracks,
                                                             m_mr.main};
    vecmem::data::vector_buffer<std::size_t> n_meas_buffer{n_tracks, m_mr.main};

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

    /*
    // Make accepted ids vector
    vecmem::data::vector_buffer<unsigned int> accepted_ids_buffer{
        n_tracks, m_mr.main, vecmem::data::buffer_type::resizable};
    vecmem::device_vector<unsigned int> accepted_ids(accepted_ids_buffer);


    // counting_iterator: 0, 1, 2, ...
    auto cit_begin = thrust::counting_iterator<int>(0);
    auto cit_end = cit_begin + n_tracks;

    // Copy the accepted ids from status vector
    auto it =
        thrust::copy_if(thrust_policy, cit_begin, cit_end, status_buffer.ptr(),
                        accepted_ids.begin(), thrust::identity<int>());

    unsigned int n_accepted =
        static_cast<unsigned int>(it - accepted_ids.begin());
    accepted_ids.resize(n_accepted);

    printf("number of accepted %d \n", n_accepted);
    */

    const unsigned int n_pre_accepted = static_cast<unsigned int>(thrust::count(
        thrust_policy, status_buffer.ptr(), status_buffer.ptr() + n_tracks, 1));

    printf("number of pre_accepted %d \n", n_pre_accepted);

    // Make accepted ids vector
    vecmem::data::vector_buffer<unsigned int> pre_accepted_ids_buffer{
        n_pre_accepted, m_mr.main};

    // counting_iterator: 0, 1, 2, ...
    auto cit_begin = thrust::counting_iterator<int>(0);
    auto cit_end = cit_begin + n_tracks;
    thrust::copy_if(thrust_policy, cit_begin, cit_end, status_buffer.ptr(),
                    pre_accepted_ids_buffer.ptr(), thrust::identity<int>());

    // Sort the flat measurement id vector
    thrust::sort(thrust_policy, flat_meas_ids_buffer.ptr(),
                 flat_meas_ids_buffer.ptr() + n_cands_total);

    // Count the number of unique measurements
    const unsigned int meas_count = static_cast<unsigned int>(
        thrust::unique_count(thrust_policy, flat_meas_ids_buffer.ptr(),
                             flat_meas_ids_buffer.ptr() + n_cands_total,
                             thrust::equal_to<int>()));
    printf("meas count %d \n", meas_count);

    // Unique measurement ids
    vecmem::data::vector_buffer<std::size_t> unique_meas_buffer{meas_count,
                                                                m_mr.main};

    // Counts of unique measurement id in flat id vector
    vecmem::data::vector_buffer<std::size_t> unique_meas_counts_buffer{
        meas_count, m_mr.main};
    m_copy.get().setup(unique_meas_counts_buffer)->ignore();

    // Counting can be done using reduce_by_key and constant iterator
    thrust::reduce_by_key(thrust_policy, flat_meas_ids_buffer.ptr(),
                          flat_meas_ids_buffer.ptr() + n_cands_total,
                          thrust::make_constant_iterator(1),
                          unique_meas_buffer.ptr(),
                          unique_meas_counts_buffer.ptr());

    // Sort unique meas ids
    thrust::sort_by_key(thrust_policy, unique_meas_buffer.ptr(),
                        unique_meas_buffer.ptr() + meas_count,
                        unique_meas_counts_buffer.ptr());

    // Retreive the counting vector to host
    std::vector<std::size_t> unique_meas_counts;
    m_copy
        .get()(unique_meas_counts_buffer, unique_meas_counts,
               vecmem::copy::type::device_to_host)
        ->wait();

    for (const auto& e : unique_meas_counts) {
        printf("unique count %lu \n", e);
    }

    // Make the tracks per measurement vector
    vecmem::data::jagged_vector_buffer<std::size_t>
        tracks_per_measurement_buffer(unique_meas_counts, m_mr.main, m_mr.host,
                                      vecmem::data::buffer_type::resizable);

    // Fill tracks per measurement vector
    {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (n_pre_accepted + nThreads - 1) / nThreads;

        kernels::fill_tracks_per_measurement<<<nBlocks, nThreads, 0, stream>>>(
            device::fill_tracks_per_measurement_payload{
                .accepted_ids_view = pre_accepted_ids_buffer,
                .meas_ids_view = meas_ids_buffer,
                .unique_meas_view = unique_meas_buffer,
                .tracks_per_measurement_view = tracks_per_measurement_buffer});
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        m_stream.get().synchronize();
    }
    /*
    // Make shared number of measurements vector
    vecmem::data::vector_buffer<unsigned int> n_shared_buffer{n_tracks,
                                                              m_mr.main};

    // Count shared number of measurements
    {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (n_pre_accepted + nThreads - 1) / nThreads;

        kernels::count_shared_measurements<<<nBlocks, nThreads, 0, stream>>>(
            device::count_shared_measurements_payload{
                .accepted_ids_view = pre_accepted_ids_buffer,
                .meas_ids_view = meas_ids_buffer,
                .unique_meas_view = unique_meas_buffer,
                .tracks_per_measurement_view = tracks_per_measurement_buffer,
                .n_shared_view = n_shared_buffer});
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        m_stream.get().synchronize();                
    }

    // Make relative shared number of measurements vector
    vecmem::data::vector_buffer<traccc::scalar> rel_shared_buffer{n_tracks,
                                                                  m_mr.main};
    */
    // Create resolved candidate buffer
    track_candidate_container_types::buffer res_candidates_buffer{
        {10, m_mr.main},
        {std::vector<std::size_t>(10, 10), m_mr.main, m_mr.host,
         vecmem::data::buffer_type::resizable}};

    return res_candidates_buffer;
}

}  // namespace traccc::cuda
