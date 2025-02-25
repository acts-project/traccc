/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// HACK: Fix for intel/llvm#15544
// As of Intel LLVM 2025.0, enabling an AMD SYCL target inadvertently sets the
// `__CUDA_ARCH__` preprocessor definition which breaks all sorts of internal
// logic in Thrust. Thus, we very selectively undefine the `__CUDA_ARCH__`
// definition when we are are compiling SYCL code using the Intel LLVM
// compiler. This can be removed when intel/llvm#15443 makes it into a OneAPI
// release.
#if defined(__INTEL_LLVM_COMPILER) && defined(SYCL_LANGUAGE_VERSION)
#undef __CUDA_ARCH__
#endif

// Project include(s).
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"
#include "traccc/fitting/status_codes.hpp"

// Detray include(s)
#include <detray/geometry/tracking_surface.hpp>

// Thrust include(s).
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace traccc::device {

template <typename detector_t, concepts::thread_id1 thread_id_t,
          concepts::barrier barrier_t>
TRACCC_DEVICE inline void find_tracks(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const finding_config& cfg, const find_tracks_payload<detector_t>& payload,
    const find_tracks_shared_payload& shared_payload) {

    /*
     * Initialize the block-shared data; in particular, set the total size of
     * the candidate buffer to zero, and then set the number of candidates for
     * each parameter to zero.
     */
    if (thread_id.getLocalThreadIdX() == 0) {
        shared_payload.shared_candidates_size = 0;
    }

    shared_payload.shared_num_candidates[thread_id.getLocalThreadIdX()] = 0;

    barrier.blockBarrier();

    /*
     * Initialize all of the device vectors from their vecmem views.
     */
    detector_t det(payload.det_data);
    measurement_collection_types::const_device measurements(
        payload.measurements_view);
    bound_track_parameters_collection_types::const_device in_params(
        payload.in_params_view);
    vecmem::device_vector<const unsigned int> in_params_liveness(
        payload.in_params_liveness_view);
    vecmem::device_vector<const candidate_link> prev_links(
        payload.prev_links_view);
    bound_track_parameters_collection_types::device out_params(
        payload.out_params_view);
    vecmem::device_vector<unsigned int> out_params_liveness(
        payload.out_params_liveness_view);
    vecmem::device_vector<candidate_link> links(payload.links_view);
    vecmem::device_atomic_ref<unsigned int,
                              vecmem::device_address_space::global>
        num_total_candidates(*payload.n_total_candidates);
    vecmem::device_vector<const detray::geometry::barcode> barcodes(
        payload.barcodes_view);
    vecmem::device_vector<const unsigned int> upper_bounds(
        payload.upper_bounds_view);

    /*
     * Compute the last step ID, using a sentinel value if the current step is
     * step 0.
     */
    const candidate_link::link_index_type::first_type previous_step =
        (payload.step == 0)
            ? std::numeric_limits<
                  candidate_link::link_index_type::first_type>::max()
            : payload.step - 1;

    const unsigned int in_param_id = thread_id.getGlobalThreadIdX();

    /*
     * Step 1 of this kernel is to determine which indices belong to which
     * parameter. Because the measurements are guaranteed to be grouped, we can
     * simply find the first measurement's index and the total number of
     * indices.
     *
     * This entire step is executed on a one-thread-one-parameter model.
     */
    unsigned int init_meas = 0;
    unsigned int num_meas = 0;

    if (in_param_id < payload.n_in_params &&
        in_params_liveness.at(in_param_id) > 0u) {
        /*
         * Get the barcode of this thread's parameters, then find the first
         * measurement that matches it.
         */
        const auto bcd = in_params.at(in_param_id).surface_link();
        const auto lo = thrust::lower_bound(thrust::seq, barcodes.begin(),
                                            barcodes.end(), bcd);

        /*
         * If we cannot find any corresponding measurements, set the number of
         * measurements to zero.
         */
        if (lo == barcodes.end()) {
            init_meas = 0;
        }
        /*
         * If measurements are found, use the previously (outside this kernel)
         * computed upper bound array to compute the range of measurements for
         * this thread.
         */
        else {
            const vecmem::device_vector<const unsigned int>::size_type bcd_id =
                static_cast<
                    vecmem::device_vector<const unsigned int>::size_type>(
                    std::distance(barcodes.begin(), lo));

            init_meas = lo == barcodes.begin() ? 0u : upper_bounds[bcd_id - 1];
            num_meas = upper_bounds[bcd_id] - init_meas;
        }
    }

    /*
     * Step 2 of this kernel involves processing the candidate measurements and
     * updating them on their corresponding surface.
     *
     * Because the number of measurements per parameter can vary wildly
     * (between 0 and 20), a naive one-thread-one-parameter model would incur a
     * lot of thread divergence here. Instead, we use a load-balanced model in
     * which threads process each others' measurements.
     *
     * The core idea is that each thread places its measurements into a shared
     * pool. We keep track of how many measurements each thread has placed into
     * the pool.
     */
    unsigned int curr_meas = 0;

    /*
     * This loop keeps running until all threads have processed all of their
     * measurements.
     */
    while (barrier.blockOr(curr_meas < num_meas ||
                           shared_payload.shared_candidates_size > 0)) {
        /*
         * The outer loop consists of three general components. The first
         * components is that each thread starts to fill a shared buffer of
         * measurements. The buffer is twice the size of the block to
         * accomodate any overflow.
         *
         * Threads insert their measurements into the shared buffer until they
         * either run out of measurements, or until the shared buffer is full.
         */
        for (; curr_meas < num_meas &&
               shared_payload.shared_candidates_size < thread_id.getBlockDimX();
             curr_meas++) {
            unsigned int idx =
                vecmem::device_atomic_ref<unsigned int,
                                          vecmem::device_address_space::local>(
                    shared_payload.shared_candidates_size)
                    .fetch_add(1u);

            /*
             * The buffer elemements are tuples of the measurement index and
             * the index of the thread that originally inserted that
             * measurement.
             */
            shared_payload.shared_candidates[idx] = {
                init_meas + curr_meas, thread_id.getLocalThreadIdX()};
        }

        barrier.blockBarrier();

        /*
         * The shared buffer is now full; each thread picks out zero or one of
         * the measurements and processes it.
         */
        if (thread_id.getLocalThreadIdX() <
            shared_payload.shared_candidates_size) {
            const unsigned int owner_local_thread_id =
                shared_payload.shared_candidates[thread_id.getLocalThreadIdX()]
                    .second;
            const unsigned int owner_global_thread_id =
                owner_local_thread_id +
                thread_id.getBlockDimX() * thread_id.getBlockIdX();
            assert(in_params_liveness.at(owner_global_thread_id) != 0u);
            const bound_track_parameters<>& in_par =
                in_params.at(owner_global_thread_id);
            const unsigned int meas_idx =
                shared_payload.shared_candidates[thread_id.getLocalThreadIdX()]
                    .first;

            const auto& meas = measurements.at(meas_idx);

            track_state<typename detector_t::algebra_type> trk_state(meas);
            const detray::tracking_surface sf{det, in_par.surface_link()};

            // Run the Kalman update
            const kalman_fitter_status res = sf.template visit_mask<
                gain_matrix_updater<typename detector_t::algebra_type>>(
                trk_state, in_par);

            const traccc::scalar chi2 = trk_state.filtered_chi2();

            // The chi2 from Kalman update should be less than chi2_max
            if (res == kalman_fitter_status::SUCCESS &&
                trk_state.filtered_chi2() < cfg.chi2_max) {
                // Add measurement candidates to link
                const unsigned int l_pos = num_total_candidates.fetch_add(1);

                assert(trk_state.filtered_chi2() >= 0.f);

                if (l_pos >= payload.n_max_candidates) {
                    *payload.n_total_candidates = payload.n_max_candidates;
                } else {
                    if (payload.step == 0) {
                        links.at(l_pos) = {
                            {previous_step, owner_global_thread_id},
                            meas_idx,
                            owner_global_thread_id,
                            0,
                            chi2};
                    } else {
                        const candidate_link& prev_link =
                            prev_links[owner_global_thread_id];

                        links.at(l_pos) = {
                            {previous_step, owner_global_thread_id},
                            meas_idx,
                            prev_link.seed_idx,
                            prev_link.n_skipped,
                            chi2};
                    }

                    // Increase the number of candidates (or branches) per input
                    // parameter
                    vecmem::device_atomic_ref<
                        unsigned int, vecmem::device_address_space::local>(
                        shared_payload
                            .shared_num_candidates[owner_local_thread_id])
                        .fetch_add(1u);

                    out_params.at(l_pos) = trk_state.filtered();
                    out_params_liveness.at(l_pos) = 1u;
                }
            }
        }

        barrier.blockBarrier();

        /*
         * The reason the buffer is twice the size of the block is that we
         * might end up having some spill-over; this spill-over should be moved
         * to the front of the buffer.
         */
        shared_payload.shared_candidates[thread_id.getLocalThreadIdX()] =
            shared_payload.shared_candidates[thread_id.getLocalThreadIdX() +
                                             thread_id.getBlockDimX()];

        if (thread_id.getLocalThreadIdX() == 0) {
            if (shared_payload.shared_candidates_size >=
                thread_id.getBlockDimX()) {
                shared_payload.shared_candidates_size -=
                    thread_id.getBlockDimX();
            } else {
                shared_payload.shared_candidates_size = 0;
            }
        }
    }

    /*
     * Part three of the kernel inserts holes for parameters which did not
     * match any measurements.
     */
    if (in_param_id < payload.n_in_params &&
        in_params_liveness.at(in_param_id) > 0u &&
        shared_payload.shared_num_candidates[thread_id.getLocalThreadIdX()] ==
            0u) {
        // Add measurement candidates to link
        const unsigned int l_pos = num_total_candidates.fetch_add(1);

        if (l_pos >= payload.n_max_candidates) {
            *payload.n_total_candidates = payload.n_max_candidates;
        } else {
            if (payload.step == 0) {
                links.at(l_pos) = {{previous_step, in_param_id},
                                   std::numeric_limits<unsigned int>::max(),
                                   in_param_id,
                                   1,
                                   std::numeric_limits<traccc::scalar>::max()};
            } else {
                const candidate_link& prev_link = prev_links[in_param_id];

                links.at(l_pos) = {{previous_step, in_param_id},
                                   std::numeric_limits<unsigned int>::max(),
                                   prev_link.seed_idx,
                                   prev_link.n_skipped + 1,
                                   std::numeric_limits<traccc::scalar>::max()};
            }

            out_params.at(l_pos) = in_params.at(in_param_id);
            out_params_liveness.at(l_pos) = 1u;
        }
    }
}

}  // namespace traccc::device
