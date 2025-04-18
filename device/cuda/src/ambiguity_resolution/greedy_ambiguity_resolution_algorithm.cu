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
#include "./kernels/update_vectors.cuh"
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

namespace traccc::cuda {

// Device operator to calculate relative number of shared measurements
struct devide_op {
    TRACCC_HOST_DEVICE
    traccc::scalar operator()(int a, int b) const {
        return static_cast<traccc::scalar>(a) / static_cast<traccc::scalar>(b);
    }
};

// Track comparator to sort the track ids
struct track_comparator {
    const traccc::scalar* rel_shared;
    const traccc::scalar* pvals;

    TRACCC_HOST_DEVICE track_comparator(const traccc::scalar* rel_shared_,
                                        const traccc::scalar* pvals_)
        : rel_shared(rel_shared_), pvals(pvals_) {}

    TRACCC_HOST_DEVICE bool operator()(unsigned int a, unsigned int b) const {
        if (rel_shared[a] != rel_shared[b]) {
            return rel_shared[a] < rel_shared[b];
        }
        return pvals[a] > pvals[b];
    }
};

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

    unsigned int n_accepted = static_cast<unsigned int>(thrust::count(
        thrust_policy, status_buffer.ptr(), status_buffer.ptr() + n_tracks, 1));

    printf("number of accepted %d \n", n_accepted);

    // Make accepted ids vector
    vecmem::data::vector_buffer<unsigned int> pre_accepted_ids_buffer{
        n_accepted, m_mr.main, vecmem::data::buffer_type::resizable};
    m_copy.get().setup(pre_accepted_ids_buffer)->ignore();
    vecmem::device_vector<unsigned int> pre_accepted_ids(
        pre_accepted_ids_buffer);
    pre_accepted_ids.resize(n_accepted);

    // Fill the accepted ids vector using counting iterator
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
    m_copy.get().setup(tracks_per_measurement_buffer)->ignore();

    // Make the number of accetped tracks per measurement vector
    vecmem::data::vector_buffer<unsigned int>
        n_accepted_tracks_per_measurement_buffer(meas_count, m_mr.main);

    // Fill tracks per measurement vector
    {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (n_accepted + nThreads - 1) / nThreads;

        kernels::fill_tracks_per_measurement<<<nBlocks, nThreads, 0, stream>>>(
            device::fill_tracks_per_measurement_payload{
                .accepted_ids_view = pre_accepted_ids_buffer,
                .meas_ids_view = meas_ids_buffer,
                .unique_meas_view = unique_meas_buffer,
                .tracks_per_measurement_view = tracks_per_measurement_buffer,
                .n_accepted_tracks_per_measurement_view =
                    n_accepted_tracks_per_measurement_buffer});
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        m_stream.get().synchronize();
    }

    // Make shared number of measurements vector
    vecmem::data::vector_buffer<unsigned int> n_shared_buffer{n_tracks,
                                                              m_mr.main};

    // Count shared number of measurements
    {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (n_accepted + nThreads - 1) / nThreads;

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

    // Fill the relative shared number of measurements vector
    thrust::transform(thrust_policy, n_shared_buffer.ptr(),
                      n_shared_buffer.ptr() + n_tracks, n_meas_buffer.ptr(),
                      rel_shared_buffer.ptr(), devide_op{});

    // Make sorted ids vector
    vecmem::data::vector_buffer<unsigned int> sorted_ids_buffer{
        n_accepted, m_mr.main, vecmem::data::buffer_type::resizable};
    m_copy.get().setup(sorted_ids_buffer)->ignore();
    vecmem::device_vector<unsigned int> sorted_ids(sorted_ids_buffer);
    sorted_ids.resize(n_accepted);

    // Fill and sort the sorted ids vector
    thrust::copy(thrust_policy, pre_accepted_ids_buffer.ptr(),
                 pre_accepted_ids_buffer.ptr() + n_accepted,
                 sorted_ids_buffer.ptr());

    track_comparator trk_comp(rel_shared_buffer.ptr(), pvals_buffer.ptr());
    thrust::sort(thrust_policy, sorted_ids_buffer.ptr(),
                 sorted_ids_buffer.ptr() + n_accepted, trk_comp);

    // Iterate over tracks
    for (unsigned int iter = 0; iter < m_config.max_iterations; iter++) {
        // Terminate if there are no tracks to iterate
        if (n_accepted == 0) {
            break;
        }

        auto max_it = thrust::max_element(thrust_policy, n_shared_buffer.ptr(),
                                          n_shared_buffer.ptr() + n_tracks);

        const unsigned int max_shared = *max_it;

        printf("Iteration: %d \n", iter);
        printf("Max shared: %d \n", max_shared);
        printf("N accepted: %d \n", n_accepted);

        // Terminate if the max shared measurements is less than the cut value
        if (max_shared < m_config.max_shared_meas) {
            break;
        }

        unsigned int worst_track;
        TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
            &worst_track, sorted_ids_buffer.ptr() + n_accepted - 1,
            sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));

        printf("worst track: %d \n", worst_track);

        int reject = 0;
        TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
            status_buffer.ptr() + worst_track, &reject, sizeof(unsigned int),
            cudaMemcpyHostToDevice, stream));

        // Fill the accepted ids vector using counting iterator
        thrust::copy_if(thrust_policy, cit_begin, cit_end, status_buffer.ptr(),
                        pre_accepted_ids_buffer.ptr(), thrust::identity<int>());

        // Remove the worst (rejected) id from the sorted ids
        n_accepted--;
        sorted_ids.resize(n_accepted);

        // Update tracks per measurement
        {
            const unsigned int nThreads = m_warp_size * 2;
            const unsigned int nBlocks =
                (meas_sizes[worst_track] + nThreads - 1) / nThreads;

            kernels::update_vectors<<<nBlocks, nThreads, 0, stream>>>(
                device::update_vectors_payload{
                    .worst_track = worst_track,
                    .meas_ids_view = meas_ids_buffer,
                    .n_meas_view = n_meas_buffer,
                    .unique_meas_view = unique_meas_buffer,
                    .tracks_per_measurement_view =
                        tracks_per_measurement_buffer,
                    .n_accepted_tracks_per_measurement_view =
                        n_accepted_tracks_per_measurement_buffer,
                    .n_shared_view = n_shared_buffer,
                    .rel_shared_view = rel_shared_buffer});
            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

            m_stream.get().synchronize();
        }

        // Keep the sorted ids vector sorted
        thrust::sort(thrust_policy, sorted_ids_buffer.ptr(),
                     sorted_ids_buffer.ptr() + n_accepted, trk_comp);
    }

    std::vector<unsigned int> accepted_ids;
    m_copy
        .get()(sorted_ids_buffer, accepted_ids,
               vecmem::copy::type::device_to_host)
        ->wait();

    std::vector<std::size_t> res_cands_size;

    printf("Final n accepted: %d \n", n_accepted);

    res_cands_size.reserve(n_accepted);
    for (unsigned int i = 0; i < n_accepted; i++) {
        const auto tid = accepted_ids[i];
        const auto n_meas = (meas_ids_buffer.ptr() + tid)->size();
        res_cands_size.push_back(n_meas);
        printf("meas size: %d \n", n_meas);
    }

    // Create resolved candidate buffer
    track_candidate_container_types::buffer res_candidates_buffer{
        {n_accepted, m_mr.main}, {res_cands_size, m_mr.main, m_mr.host}};

    return res_candidates_buffer;
}

}  // namespace traccc::cuda
