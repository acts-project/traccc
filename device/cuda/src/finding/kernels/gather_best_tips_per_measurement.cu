/** traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "gather_best_tips_per_measurement.cuh"

namespace traccc::cuda::kernels {

__global__ void gather_best_tips_per_measurement(
    const vecmem::data::vector_view<const unsigned int> tips_view,
    const vecmem::data::vector_view<const candidate_link> links_view,
    const edm::measurement_collection<default_algebra>::const_view
        measurements_view,
    vecmem::data::vector_view<unsigned long long int> insertion_mutex_view,
    vecmem::data::vector_view<unsigned int> tip_index_view,
    vecmem::data::vector_view<scalar> tip_pval_view,
    const unsigned int max_num_tracks_per_measurement) {
    unsigned int tip_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const vecmem::device_vector<const unsigned int> tips(tips_view);
    const vecmem::device_vector<const candidate_link> links(links_view);
    const edm::measurement_collection<default_algebra>::const_device
        measurements(measurements_view);
    vecmem::device_vector<unsigned long long int> insertion_mutex(
        insertion_mutex_view);
    vecmem::device_vector<unsigned int> tip_index(tip_index_view);
    vecmem::device_vector<scalar> tip_pval(tip_pval_view);
    const unsigned int n_meas = measurements.size();

    scalar pval = 0.f;
    unsigned int link_idx = 0;
    unsigned int num_states = 0;

    bool need_to_write = true;
    candidate_link L;

    if (tip_idx < tips.size()) {
        link_idx = tips.at(tip_idx);
        const auto link = links.at(link_idx);
        pval = prob(link.chi2_sum, static_cast<scalar>(link.ndf_sum) - 5.f);
        num_states = link.step + 1 - link.n_skipped;

        L = link;

        // Skip any holes at the start; there shouldn't be any.
        while (L.meas_idx >= n_meas && L.step != 0u) {
            L = links.at(L.previous_candidate_idx);
        }
    } else {
        need_to_write = false;
    }

    unsigned int current_state = 0;

    while (__syncthreads_or(current_state < num_states || need_to_write)) {
        if (current_state < num_states || need_to_write) {
            assert(L.meas_idx < n_meas);

            if (need_to_write) {
                vecmem::device_atomic_ref<unsigned long long int> mutex(
                    insertion_mutex.at(L.meas_idx));

                unsigned long long int assumed = mutex.load();
                unsigned long long int desired_set;
                auto [locked, size, worst] =
                    device::decode_insertion_mutex(assumed);

                if (need_to_write && size >= max_num_tracks_per_measurement &&
                    pval <= worst) {
                    need_to_write = false;
                }

                bool holds_lock = false;

                if (need_to_write && !locked) {
                    desired_set =
                        device::encode_insertion_mutex(true, size, worst);

                    if (mutex.compare_exchange_strong(assumed, desired_set)) {
                        holds_lock = true;
                    }
                }

                if (holds_lock) {
                    unsigned int new_size;
                    unsigned int offset =
                        L.meas_idx * max_num_tracks_per_measurement;
                    unsigned int out_idx;

                    if (size == max_num_tracks_per_measurement) {
                        new_size = size;

                        scalar worst_pval = std::numeric_limits<scalar>::max();

                        for (unsigned int i = 0; i < size; ++i) {
                            if (tip_pval.at(offset + i) < worst_pval) {
                                worst_pval = tip_pval.at(offset + i);
                                out_idx = i;
                            }
                        }
                    } else {
                        new_size = size + 1;
                        out_idx = size;
                    }

                    tip_index.at(offset + out_idx) = tip_idx;
                    tip_pval.at(offset + out_idx) = pval;

                    scalar new_worst = std::numeric_limits<scalar>::max();

                    for (unsigned int i = 0; i < new_size; ++i) {
                        new_worst =
                            std::min(new_worst, tip_pval.at(offset + i));
                    }

                    [[maybe_unused]] bool cas_result =
                        mutex.compare_exchange_strong(
                            desired_set, device::encode_insertion_mutex(
                                             false, new_size, new_worst));

                    assert(cas_result);

                    need_to_write = false;
                }
            }

            if (!need_to_write) {
                if (current_state < num_states - 1) {
                    L = links.at(L.previous_candidate_idx);
                    while (L.meas_idx >= n_meas && L.step != 0u) {
                        L = links.at(L.previous_candidate_idx);
                    }
                    need_to_write = true;
                } else {
#ifndef NDEBUG
                    if (L.step != 0) {
                        do {
                            L = links.at(L.previous_candidate_idx);
                        } while (L.meas_idx >= n_meas && L.step != 0u);
                        assert(L.meas_idx >= n_meas);
                    }
                    assert(L.step == 0);
#endif
                }

                current_state++;
            }
        }
    }
}

}  // namespace traccc::cuda::kernels
