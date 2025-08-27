/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/utils.hpp"
#include "./kernels/add_block_offset.cuh"
#include "./kernels/block_inclusive_scan.cuh"
#include "./kernels/count_shared_measurements.cuh"
#include "./kernels/fill_inverted_ids.cuh"
#include "./kernels/fill_track_candidates.cuh"
#include "./kernels/fill_tracks_per_measurement.cuh"
#include "./kernels/fill_unique_meas_id_map.cuh"
#include "./kernels/fill_vectors.cuh"
#include "./kernels/rearrange_tracks.cuh"
#include "./kernels/remove_tracks.cuh"
#include "./kernels/scan_block_offsets.cuh"
#include "./kernels/sort_tracks_per_measurement.cuh"
#include "./kernels/sort_updated_tracks.cuh"
#include "./kernels/update_status.cuh"
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/definitions/math.hpp"

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
namespace traccc::cuda {

struct identity_op {
    template <typename T>
    TRACCC_HOST_DEVICE T operator()(T i) const {
        return i;
    }
};

// Device operator to calculate relative number of shared measurements
struct devide_op {
    TRACCC_HOST_DEVICE
    traccc::scalar operator()(unsigned int a, unsigned int b) const {
        return math::div_ieee754(static_cast<traccc::scalar>(a),
                                 static_cast<traccc::scalar>(b));
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

struct measurement_id_comparator {
    TRACCC_HOST_DEVICE bool operator()(const measurement& a,
                                       const measurement& b) const {
        return a.measurement_id < b.measurement_id;
    }
};

greedy_ambiguity_resolution_algorithm::greedy_ambiguity_resolution_algorithm(
    const config_type& cfg, const traccc::memory_resource& mr,
    vecmem::copy& copy, stream& str, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_config(cfg),
      m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_warp_size(details::get_warp_size(str.device())) {}

greedy_ambiguity_resolution_algorithm::output_type
greedy_ambiguity_resolution_algorithm::operator()(
    const edm::track_candidate_container<default_algebra>::const_view&
        track_candidates_view) const {

    measurement_collection_types::const_device measurements(
        track_candidates_view.measurements);

    auto n_meas_total =
        m_copy.get().get_size(track_candidates_view.measurements);

    // Make sure that max_measurement_id = number_of_measurement -1
    // @TODO: More robust way is to assert that measurement id ranges from 0, 1,
    // ..., number_of_measurement - 1
    [[maybe_unused]] auto max_meas_it = thrust::max_element(
        thrust::device, track_candidates_view.measurements.ptr(),
        track_candidates_view.measurements.ptr() + n_meas_total,
        measurement_id_comparator{});

    measurement max_meas;
    cudaMemcpy(&max_meas, thrust::raw_pointer_cast(&(*max_meas_it)),
               sizeof(measurement), cudaMemcpyDeviceToHost);

    if (max_meas.measurement_id != n_meas_total - 1) {
        throw std::runtime_error(
            "max measurement id should be equal to (the number of measurements "
            "- 1)");
    }

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // The Thrust policy to use.
    auto thrust_policy =
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(m_mr.main)))
            .on(stream);

    const unsigned int n_tracks = track_candidates_view.tracks.capacity();

    if (n_tracks == 0) {
        return {};
    }

    // Make sure that max_shared_meas is largen than zero
    assert(m_config.max_shared_meas > 0u);

    // Status (1 = Accept, 0 = Reject)
    vecmem::data::vector_buffer<int> status_buffer{n_tracks, m_mr.main};

    vecmem::device_vector<int> status_device(status_buffer);
    thrust::fill(thrust_policy, status_device.begin(), status_device.end(), 1);

    // Get the sizes of the measurement index vector in each track
    const std::vector<unsigned int> candidate_sizes =
        m_copy.get().get_sizes(track_candidates_view.tracks);

    // Make measurement ID, pval and n_measurement vector
    vecmem::data::jagged_vector_buffer<measurement_id_type> meas_ids_buffer{
        candidate_sizes, m_mr.main, m_mr.host,
        vecmem::data::buffer_type::resizable};
    m_copy.get().setup(meas_ids_buffer)->ignore();

    const unsigned int n_cands_total =
        std::accumulate(candidate_sizes.begin(), candidate_sizes.end(), 0u);

    vecmem::data::vector_buffer<measurement_id_type> flat_meas_ids_buffer{
        n_cands_total, m_mr.main, vecmem::data::buffer_type::resizable};
    m_copy.get().setup(flat_meas_ids_buffer)->ignore();
    vecmem::data::vector_buffer<traccc::scalar> pvals_buffer{n_tracks,
                                                             m_mr.main};
    vecmem::data::vector_buffer<unsigned int> n_meas_buffer{n_tracks,
                                                            m_mr.main};
    thrust::fill(thrust_policy, n_meas_buffer.ptr(),
                 n_meas_buffer.ptr() + n_tracks, 0);

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

    vecmem::unique_alloc_ptr<unsigned int> n_accepted_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);
    TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(n_accepted_device.get(),
                                            &n_accepted, sizeof(unsigned int),
                                            cudaMemcpyHostToDevice, stream));

    m_stream.get().synchronize();

    if (n_accepted == 0) {
        return {};
    }

    // Make accepted ids vector
    vecmem::data::vector_buffer<unsigned int> pre_accepted_ids_buffer{
        n_accepted, m_mr.main};

    m_copy.get().setup(pre_accepted_ids_buffer)->ignore();

    // Fill the accepted ids vector using counting iterator
    auto cit_begin = thrust::counting_iterator<int>(0);
    auto cit_end = cit_begin + n_tracks;
    thrust::copy_if(thrust_policy, cit_begin, cit_end, status_buffer.ptr(),
                    pre_accepted_ids_buffer.ptr(), identity_op{});

    // Sort the flat measurement id vector
    thrust::sort(thrust_policy, flat_meas_ids_buffer.ptr(),
                 flat_meas_ids_buffer.ptr() + n_cands_total);

    // Count the number of unique measurements
    const unsigned int meas_count = static_cast<unsigned int>(
        thrust::unique_count(thrust_policy, flat_meas_ids_buffer.ptr(),
                             flat_meas_ids_buffer.ptr() + n_cands_total,
                             thrust::equal_to<int>()));

    // Unique measurement ids
    vecmem::data::vector_buffer<measurement_id_type> unique_meas_buffer{
        meas_count, m_mr.main};

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

    // Unique measurement ids
    vecmem::data::vector_buffer<measurement_id_type>
        meas_id_to_unique_id_buffer{max_meas.measurement_id + 1, m_mr.main};

    // Make meas_id to meas vector
    {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (meas_count + nThreads - 1) / nThreads;

        kernels::fill_unique_meas_id_map<<<nBlocks, nThreads, 0, stream>>>(
            device::fill_unique_meas_id_map_payload{
                .unique_meas_view = unique_meas_buffer,
                .meas_id_to_unique_id_view = meas_id_to_unique_id_buffer});
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        m_stream.get().synchronize();
    }

    // Retreive the counting vector to host
    std::vector<std::size_t> unique_meas_counts;
    m_copy
        .get()(unique_meas_counts_buffer, unique_meas_counts,
               vecmem::copy::type::device_to_host)
        ->wait();

    // Make the tracks per measurement vector
    vecmem::data::jagged_vector_buffer<unsigned int>
        tracks_per_measurement_buffer(unique_meas_counts, m_mr.main, m_mr.host,
                                      vecmem::data::buffer_type::resizable);
    m_copy.get().setup(tracks_per_measurement_buffer)->ignore();

    // Make the track status per measurement vector
    vecmem::data::jagged_vector_buffer<int> track_status_per_measurement_buffer(
        unique_meas_counts, m_mr.main, m_mr.host,
        vecmem::data::buffer_type::resizable);

    m_copy.get().setup(track_status_per_measurement_buffer)->ignore();

    // Make the number of accetped tracks per measurement vector
    vecmem::data::vector_buffer<unsigned int>
        n_accepted_tracks_per_measurement_buffer(meas_count, m_mr.main);
    thrust::fill(thrust_policy, n_accepted_tracks_per_measurement_buffer.ptr(),
                 n_accepted_tracks_per_measurement_buffer.ptr() + meas_count,
                 0);

    // Fill tracks per measurement vector
    {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (n_accepted + nThreads - 1) / nThreads;

        kernels::fill_tracks_per_measurement<<<nBlocks, nThreads, 0, stream>>>(
            device::fill_tracks_per_measurement_payload{
                .accepted_ids_view = pre_accepted_ids_buffer,
                .meas_ids_view = meas_ids_buffer,
                .meas_id_to_unique_id_view = meas_id_to_unique_id_buffer,
                .tracks_per_measurement_view = tracks_per_measurement_buffer,
                .track_status_per_measurement_view =
                    track_status_per_measurement_buffer,
                .n_accepted_tracks_per_measurement_view =
                    n_accepted_tracks_per_measurement_buffer});
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        m_stream.get().synchronize();
    }

    // Sort tracks per measurement vector
    // @TODO: For the case where the measurement is shared by more than 1024
    // tracks, the tracks need to be sorted again using thrust::sort
    {
        const unsigned int nThreads = 1024;
        const unsigned int nBlocks = meas_count;

        kernels::sort_tracks_per_measurement<<<nBlocks, nThreads, 0, stream>>>(
            device::sort_tracks_per_measurement_payload{
                .tracks_per_measurement_view = tracks_per_measurement_buffer,
            });
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        m_stream.get().synchronize();
    }

    // Make shared number of measurements vector
    vecmem::data::vector_buffer<unsigned int> n_shared_buffer{n_tracks,
                                                              m_mr.main};
    thrust::fill(thrust_policy, n_shared_buffer.ptr(),
                 n_shared_buffer.ptr() + n_tracks, 0);
    m_copy.get().setup(n_shared_buffer)->ignore();

    // Count shared number of measurements
    {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (n_accepted + nThreads - 1) / nThreads;

        kernels::count_shared_measurements<<<nBlocks, nThreads, 0, stream>>>(
            device::count_shared_measurements_payload{
                .accepted_ids_view = pre_accepted_ids_buffer,
                .meas_ids_view = meas_ids_buffer,
                .meas_id_to_unique_id_view = meas_id_to_unique_id_buffer,
                .n_accepted_tracks_per_measurement_view =
                    n_accepted_tracks_per_measurement_buffer,
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
    vecmem::data::vector_buffer<unsigned int> sorted_ids_buffer{n_accepted,
                                                                m_mr.main};
    m_copy.get().setup(sorted_ids_buffer)->ignore();
    vecmem::data::vector_buffer<unsigned int> temp_sorted_ids_buffer{n_accepted,
                                                                     m_mr.main};
    m_copy.get().setup(temp_sorted_ids_buffer)->ignore();

    // track id to the index of sorted ids
    vecmem::data::vector_buffer<unsigned int> inverted_ids_buffer{n_tracks,
                                                                  m_mr.main};
    m_copy.get().setup(inverted_ids_buffer)->ignore();

    // Whether track id is updated after an iteration
    vecmem::data::vector_buffer<int> is_updated_buffer{n_tracks, m_mr.main};
    m_copy.get().setup(is_updated_buffer)->ignore();
    m_copy.get().memset(is_updated_buffer, 0)->ignore();

    // Count track id apperance during removal process
    vecmem::data::vector_buffer<int> track_count_buffer{n_tracks, m_mr.main};
    m_copy.get().setup(track_count_buffer)->ignore();
    m_copy.get().memset(track_count_buffer, 0)->ignore();

    // Prefix sum buffer
    vecmem::data::vector_buffer<int> prefix_sums_buffer{n_tracks, m_mr.main};
    m_copy.get().setup(prefix_sums_buffer)->ignore();

    // Fill and sort the sorted ids vector
    thrust::copy(thrust_policy, pre_accepted_ids_buffer.ptr(),
                 pre_accepted_ids_buffer.ptr() + n_accepted,
                 sorted_ids_buffer.ptr());
    m_stream.get().synchronize();

    track_comparator trk_comp(rel_shared_buffer.ptr(), pvals_buffer.ptr());
    thrust::sort(thrust_policy, sorted_ids_buffer.ptr(),
                 sorted_ids_buffer.ptr() + n_accepted, trk_comp);

    // Update track ids
    vecmem::data::vector_buffer<unsigned int> updated_tracks_buffer{n_accepted,
                                                                    m_mr.main};
    m_copy.get().setup(updated_tracks_buffer)->ignore();

    // Device objects
    vecmem::unique_alloc_ptr<unsigned int> n_removable_tracks_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);
    vecmem::unique_alloc_ptr<unsigned int> n_meas_to_remove_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);
    vecmem::unique_alloc_ptr<unsigned int> n_valid_threads_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);

    int terminate = 0;
    vecmem::unique_alloc_ptr<int> terminate_device =
        vecmem::make_unique_alloc<int>(m_mr.main);
    auto max_shared = thrust::max_element(thrust::device, n_shared_buffer.ptr(),
                                          n_shared_buffer.ptr() + n_tracks);
    vecmem::unique_alloc_ptr<unsigned int> max_shared_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);
    cudaMemcpyAsync(max_shared_device.get(), max_shared, sizeof(unsigned int),
                    cudaMemcpyHostToDevice, stream);

    vecmem::unique_alloc_ptr<unsigned int> n_updated_tracks_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);

    // Thread block size
    unsigned int nThreads_adaptive = m_warp_size;
    unsigned int nBlocks_adaptive =
        (n_accepted + nThreads_adaptive - 1) / nThreads_adaptive;

    unsigned int nThreads_rearrange = 1024;
    unsigned int nBlocks_rearrange =
        (n_accepted + (nThreads_rearrange / kernels::nThreads_per_track) - 1) /
        (nThreads_rearrange / kernels::nThreads_per_track);

    // Compute the threadblock dimension for scanning kernels
    auto compute_scan_config = [&](unsigned int n_accepted) {
        unsigned int nThreads_scan = m_warp_size * 4;
        unsigned int nBlocks_scan =
            (n_accepted + nThreads_scan - 1) / nThreads_scan;

        while (nThreads_scan <= 1024) {
            if (nBlocks_scan > 1024) {
                nThreads_scan *= 2;
                nBlocks_scan = (n_accepted + nThreads_scan - 1) / nThreads_scan;
            } else {
                break;
            }
        }

        return std::make_pair(nThreads_scan, nBlocks_scan);
    };

    auto scan_dim = compute_scan_config(n_accepted);
    unsigned int nThreads_scan = scan_dim.first;
    unsigned int nBlocks_scan = scan_dim.second;

    assert(nBlocks_scan <= 1024 &&
           "nBlocks_scan larger than 1024 will cause invalid arguments in "
           "scan_block_offsets kernel");

    // block offsets buffer
    vecmem::data::vector_buffer<int> block_offsets_buffer{nBlocks_scan,
                                                          m_mr.main};
    m_copy.get().setup(block_offsets_buffer)->ignore();
    vecmem::data::vector_buffer<int> scanned_block_offsets_buffer{nBlocks_scan,
                                                                  m_mr.main};
    m_copy.get().setup(block_offsets_buffer)->ignore();

    while (!terminate && n_accepted > 0) {
        nBlocks_adaptive =
            (n_accepted + nThreads_adaptive - 1) / nThreads_adaptive;

        scan_dim = compute_scan_config(n_accepted);
        nThreads_scan = scan_dim.first;
        nBlocks_scan = scan_dim.second;
        nBlocks_rearrange =
            (n_accepted + (nThreads_rearrange / kernels::nThreads_per_track) -
             1) /
            (nThreads_rearrange / kernels::nThreads_per_track);

        // Make CUDA Graph
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        kernels::remove_tracks<<<1, 512, 0, stream>>>(
            device::remove_tracks_payload{
                .sorted_ids_view = sorted_ids_buffer,
                .n_accepted = n_accepted_device.get(),
                .meas_ids_view = meas_ids_buffer,
                .n_meas_view = n_meas_buffer,
                .meas_id_to_unique_id_view = meas_id_to_unique_id_buffer,
                .tracks_per_measurement_view = tracks_per_measurement_buffer,
                .track_status_per_measurement_view =
                    track_status_per_measurement_buffer,
                .n_accepted_tracks_per_measurement_view =
                    n_accepted_tracks_per_measurement_buffer,
                .n_shared_view = n_shared_buffer,
                .rel_shared_view = rel_shared_buffer,
                .n_removable_tracks = n_removable_tracks_device.get(),
                .n_meas_to_remove = n_meas_to_remove_device.get(),
                .terminate = terminate_device.get(),
                .max_shared = max_shared_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .updated_tracks_view = updated_tracks_buffer,
                .is_updated_view = is_updated_buffer,
                .n_valid_threads = n_valid_threads_device.get(),
                .track_count_view = track_count_buffer});

        // The seven kernels below are to keep sorted_ids sorted based on
        // the relative shared measurements and pvalues. This can be reduced
        // into thrust::sort():
        /*
        cudaMemcpyAsync(&n_accepted, n_accepted_device.get(),
                        sizeof(unsigned int), cudaMemcpyDeviceToHost,
        stream); thrust::sort(thrust_policy, sorted_ids_buffer.ptr(),
                     sorted_ids_buffer.ptr() + n_accepted,
                     trk_comp);
        */
        // Disadvantage: we need to do device-host copy which has large
        // overhead and CUDA graph is not available anymore
        // Advantage: This works for all cases (The below kernels only work
        // when the number of updated tracks <= 1024) and might be faster
        // with large number of updated tracks

        kernels::sort_updated_tracks<<<1, 512, 0, stream>>>(
            device::sort_updated_tracks_payload{
                .rel_shared_view = rel_shared_buffer,
                .pvals_view = pvals_buffer,
                .terminate = terminate_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .updated_tracks_view = updated_tracks_buffer,
            });

        kernels::fill_inverted_ids<<<nBlocks_adaptive, nThreads_adaptive, 0,
                                     stream>>>(
            device::fill_inverted_ids_payload{
                .sorted_ids_view = sorted_ids_buffer,
                .terminate = terminate_device.get(),
                .n_accepted = n_accepted_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .inverted_ids_view = inverted_ids_buffer,
            });

        kernels::block_inclusive_scan<<<nBlocks_scan, nThreads_scan,
                                        nThreads_scan * sizeof(int), stream>>>(
            device::block_inclusive_scan_payload{
                .sorted_ids_view = sorted_ids_buffer,
                .terminate = terminate_device.get(),
                .n_accepted = n_accepted_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .is_updated_view = is_updated_buffer,
                .block_offsets_view = block_offsets_buffer,
                .prefix_sums_view = prefix_sums_buffer});

        kernels::scan_block_offsets<<<1, nBlocks_scan,
                                      nBlocks_scan * sizeof(int), stream>>>(
            device::scan_block_offsets_payload{
                .terminate = terminate_device.get(),
                .n_accepted = n_accepted_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .block_offsets_view = block_offsets_buffer,
                .scanned_block_offsets_view = scanned_block_offsets_buffer});

        kernels::add_block_offset<<<nBlocks_scan, nThreads_scan, 0, stream>>>(
            device::add_block_offset_payload{
                .terminate = terminate_device.get(),
                .n_accepted = n_accepted_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .block_offsets_view = scanned_block_offsets_buffer,
                .prefix_sums_view = prefix_sums_buffer});

        kernels::rearrange_tracks<<<nBlocks_rearrange, nThreads_rearrange, 0,
                                    stream>>>(device::rearrange_tracks_payload{
            .sorted_ids_view = sorted_ids_buffer,
            .inverted_ids_view = inverted_ids_buffer,
            .rel_shared_view = rel_shared_buffer,
            .pvals_view = pvals_buffer,
            .terminate = terminate_device.get(),
            .n_accepted = n_accepted_device.get(),
            .n_updated_tracks = n_updated_tracks_device.get(),
            .updated_tracks_view = updated_tracks_buffer,
            .is_updated_view = is_updated_buffer,
            .prefix_sums_view = prefix_sums_buffer,
            .temp_sorted_ids_view = temp_sorted_ids_buffer,
        });

        kernels::
            update_status<<<nBlocks_adaptive, nThreads_adaptive, 0, stream>>>(
                device::update_status_payload{
                    .terminate = terminate_device.get(),
                    .n_accepted = n_accepted_device.get(),
                    .n_updated_tracks = n_updated_tracks_device.get(),
                    .temp_sorted_ids_view = temp_sorted_ids_buffer,
                    .sorted_ids_view = sorted_ids_buffer,
                    .updated_tracks_view = updated_tracks_buffer,
                    .is_updated_view = is_updated_buffer,
                    .n_shared_view = n_shared_buffer,
                    .max_shared = max_shared_device.get()});

        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

        // TODO: Make n_it adaptive based on the average track length, bound
        // value in remove_tracks, etc.
        const unsigned int n_it = 100;
        for (unsigned int iter = 0; iter < n_it; iter++) {
            cudaGraphLaunch(graphExec, stream);
        }

        cudaMemcpyAsync(&terminate, terminate_device.get(), sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&n_accepted, n_accepted_device.get(),
                        sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    }

    cudaMemcpyAsync(&n_accepted, n_accepted_device.get(), sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, stream);

    auto max_it =
        std::max_element(candidate_sizes.begin(), candidate_sizes.end());
    const unsigned int max_cands_size = *max_it;

    // Create resolved candidate buffer
    edm::track_candidate_collection<default_algebra>::buffer
        res_track_candidates_buffer{
            std::vector<std::size_t>(n_accepted, max_cands_size), m_mr.main,
            m_mr.host, vecmem::data::buffer_type::resizable};
    m_copy.get().setup(res_track_candidates_buffer)->ignore();

    // Fill the output track candidates
    {
        if (n_accepted > 0) {
            kernels::fill_track_candidates<<<
                static_cast<unsigned int>((n_accepted + 63) / 64), 64, 0,
                stream>>>(device::fill_track_candidates_payload{
                .track_candidates_view = track_candidates_view.tracks,
                .n_accepted = n_accepted,
                .sorted_ids_view = sorted_ids_buffer,
                .res_track_candidates_view = res_track_candidates_buffer});
            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

            m_stream.get().synchronize();
        }
    }

    return res_track_candidates_buffer;
}

}  // namespace traccc::cuda
