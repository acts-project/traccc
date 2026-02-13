/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/finding/device/combinatorial_kalman_filter_algorithm.hpp"

// Project include(s).
#include "traccc/finding/candidate_link.hpp"

// System include(s).
#include <stdexcept>

namespace traccc::device {

struct combinatorial_kalman_filter_algorithm::data {
    /// The (finding) algorithm configuration
    finding_config m_config;
};

combinatorial_kalman_filter_algorithm::combinatorial_kalman_filter_algorithm(
    const finding_config& config, const traccc::memory_resource& mr,
    vecmem::copy& copy, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      algorithm_base{mr, copy},
      m_data{std::make_unique<data>(config)} {

    if (config.min_step_length_for_next_surface <=
        math::fabs(
            config.propagation.navigation.intersection.overstep_tolerance)) {
        throw std::invalid_argument(
            "Min step length for the next surface should be higher than the "
            "overstep tolerance");
    }
}

combinatorial_kalman_filter_algorithm::
    ~combinatorial_kalman_filter_algorithm() = default;

auto combinatorial_kalman_filter_algorithm::operator()(
    const detector_buffer& det, const magnetic_field& bfield,
    const edm::measurement_collection<default_algebra>::const_view&
        measurements_view,
    const bound_track_parameters_collection_types::const_view& seeds) const
    -> output_type {

    assert(m_data);
    assert(input_is_valid(measurements_view));

    //  const edm::measurement_collection<default_algebra>::const_device
    //      measurements{measurements_view};

    /*****************************************************************
     * Measurement Operations
     *****************************************************************/

    // Get the number of measurements. In an asynchronous way if possible.
    edm::measurement_collection<default_algebra>::const_view::size_type
        n_measurements = 0u;
    if (mr().host) {
        vecmem::async_size size =
            copy().get_size(measurements_view, *(mr().host));
        // Here we could give control back to the caller, once our code allows
        // for it. (coroutines...)
        n_measurements = size.get();
    } else {
        n_measurements = copy().get_size(measurements_view);
    }

    // Get upper bounds of measurement ranges per surface
    vecmem::data::vector_buffer<unsigned int> meas_ranges_buffer =
        build_measurement_ranges_buffer(det, n_measurements, measurements_view);

    // Get the number of track seeds. In an asynchronous way if possible.
    bound_track_parameters_collection_types::const_view::size_type n_seeds = 0u;
    if (mr().host) {
        vecmem::async_size size = copy().get_size(seeds, *(mr().host));
        // Here we could give control back to the caller, once our code allows
        // for it. (coroutines...)
        n_seeds = size.get();
    } else {
        n_seeds = copy().get_size(seeds);
    }

    // Prepare input parameters with seeds
    bound_track_parameters_collection_types::buffer in_params_buffer(n_seeds,
                                                                     mr().main);
    copy().setup(in_params_buffer)->ignore();
    copy()(seeds, in_params_buffer, vecmem::copy::type::device_to_device)
        ->ignore();
    vecmem::data::vector_buffer<unsigned int> param_liveness_buffer(n_seeds,
                                                                    mr().main);
    copy().setup(param_liveness_buffer)->ignore();
    copy().memset(param_liveness_buffer, 1)->ignore();

    // Number of tracks per seed
    vecmem::data::vector_buffer<unsigned int> n_tracks_per_seed_buffer(
        n_seeds, mr().main);
    copy().setup(n_tracks_per_seed_buffer)->ignore();

    // Compute the effective number of initial links per seed. If the
    // branching factor (`max_num_branches_per_surface`) is arbitrary there
    // is no useful upper bound on the number of links, but if the branching
    // factor is exactly one, we can never have more links per seed than the
    // number of CKF steps, which is a useful upper bound.
    const unsigned int effective_initial_links_per_seed =
        m_data->m_config.max_num_branches_per_surface == 1
            ? std::min(m_data->m_config.initial_links_per_seed,
                       m_data->m_config.max_track_candidates_per_track)
            : m_data->m_config.initial_links_per_seed;

    // Create a buffer of candidate links
    unsigned int link_buffer_capacity =
        effective_initial_links_per_seed * n_seeds;
    vecmem::data::vector_buffer<candidate_link> links_buffer(
        link_buffer_capacity, mr().main, vecmem::data::buffer_type::resizable);
    copy().setup(links_buffer)->ignore();

    // Buffers needed for MBF smoother (if enabled).
    vecmem::data::vector_buffer<bound_matrix<default_algebra>> jacobian_buffer,
        tmp_jacobian_buffer;
    bound_track_parameters_collection_types::buffer
        link_predicted_parameter_buffer,
        link_filtered_parameter_buffer;

    /*
     * If we are aiming to run the MBF smoother at the end of the track
     * finding, we need some space to store the intermediate Jacobians
     * and parameters. Allocate that space here.
     */
    if (m_data->m_config.run_mbf_smoother) {
        jacobian_buffer = {link_buffer_capacity, mr().main};
        link_predicted_parameter_buffer = {link_buffer_capacity, mr().main};
        link_filtered_parameter_buffer = {link_buffer_capacity, mr().main};
    }

    // Create a buffer of tip links
    const unsigned int tips_buffer_capacity =
        m_data->m_config.max_num_branches_per_seed * n_seeds;
    vecmem::data::vector_buffer<unsigned int> tips_buffer{
        tips_buffer_capacity, mr().main, vecmem::data::buffer_type::resizable};
    copy().setup(tips_buffer)->ignore();
    vecmem::data::vector_buffer<unsigned int> tip_length_buffer{
        tips_buffer_capacity, mr().main};
    copy().setup(tip_length_buffer)->ignore();
    copy().memset(tip_length_buffer, 0)->ignore();

    std::map<unsigned int, unsigned int> step_to_link_idx_map;
    step_to_link_idx_map[0u] = 0u;

    unsigned int n_in_params = n_seeds;
    for (unsigned int step = 0;
         step < m_data->m_config.max_track_candidates_per_track &&
         n_in_params > 0;
         ++step) {

        /*****************************************************************
         * Apply material interaction
         ****************************************************************/
        apply_interaction_kernel(
            n_in_params, m_data->m_config, det,
            {.n_params = n_in_params,
             .params_view = in_params_buffer,
             .params_liveness_view = param_liveness_buffer});

        /*****************************************************************
         * Find valid tracks
         *****************************************************************/

        unsigned int n_candidates = 0;

        // Buffer for kalman-updated parameters spawned by the
        // measurement candidates
        const unsigned int n_max_candidates =
            n_in_params * m_data->m_config.max_num_branches_per_surface;

        bound_track_parameters_collection_types::buffer updated_params_buffer(
            n_max_candidates, mr().main);
        copy().setup(updated_params_buffer)->ignore();
        vecmem::data::vector_buffer<unsigned int> updated_liveness_buffer(
            n_max_candidates, mr().main);
        copy().setup(updated_liveness_buffer)->ignore();

        // Reset the number of tracks per seed
        copy().memset(n_tracks_per_seed_buffer, 0)->ignore();

        const unsigned int links_size = step_to_link_idx_map[step];

        // Ensure that the links buffer is large enough to hold all new links.
        if (links_size + n_max_candidates > link_buffer_capacity) {

            const unsigned int new_link_buffer_capacity = std::max(
                2 * link_buffer_capacity, links_size + n_max_candidates);
            TRACCC_INFO("Link buffer (capacity "
                        << link_buffer_capacity << ") is too small to hold "
                        << links_size << " current and " << n_max_candidates
                        << " new links; increasing capacity to "
                        << new_link_buffer_capacity);

            link_buffer_capacity = new_link_buffer_capacity;

            vecmem::data::vector_buffer<candidate_link> new_links_buffer(
                link_buffer_capacity, mr().main,
                vecmem::data::buffer_type::resizable);
            copy().setup(new_links_buffer)->ignore();
            copy()(links_buffer, new_links_buffer)->wait();

            links_buffer = std::move(new_links_buffer);

            if (m_data->m_config.run_mbf_smoother) {

                // Create new, larger buffers for the MBF smoother data.
                vecmem::data::vector_buffer<bound_matrix<default_algebra>>
                    new_jacobian_buffer{link_buffer_capacity, mr().main};
                bound_track_parameters_collection_types::buffer
                    new_link_predicted_parameter_buffer{link_buffer_capacity,
                                                        mr().main},
                    new_link_filtered_parameter_buffer{link_buffer_capacity,
                                                       mr().main};

                // Copy old data to new buffers.
                vecmem::copy::event_type ev1 =
                    copy()(jacobian_buffer, new_jacobian_buffer);
                vecmem::copy::event_type ev2 =
                    copy()(link_predicted_parameter_buffer,
                           new_link_predicted_parameter_buffer);
                vecmem::copy::event_type ev3 =
                    copy()(link_filtered_parameter_buffer,
                           new_link_filtered_parameter_buffer);

                // Here we could give control back to the caller, once our code
                // allows for it. (coroutines...)
                ev1->wait();
                ev2->wait();
                ev3->wait();

                // Replace old buffers with the new ones.
                jacobian_buffer = std::move(new_jacobian_buffer);
                link_predicted_parameter_buffer =
                    std::move(new_link_predicted_parameter_buffer);
                link_filtered_parameter_buffer =
                    std::move(new_link_filtered_parameter_buffer);
            }
        }

        {
            vecmem::data::vector_buffer<candidate_link> tmp_links_buffer(
                n_max_candidates, mr().main);
            copy().setup(tmp_links_buffer)->ignore();
            bound_track_parameters_collection_types::buffer tmp_params_buffer(
                n_max_candidates, mr().main);
            copy().setup(tmp_params_buffer)->ignore();

            // Launch the kernel.
            find_tracks_kernel(
                n_in_params,
                {.config = m_data->m_config,
                 .n_params = n_in_params,
                 .det = det,
                 .measurements = measurements_view,
                 .in_params = in_params_buffer,
                 .in_params_liveness = param_liveness_buffer,
                 .measurement_ranges = meas_ranges_buffer,
                 .links = links_buffer,
                 .prev_links_idx =
                     (step == 0 ? 0 : step_to_link_idx_map[step - 1]),
                 .curr_links_idx = step_to_link_idx_map[step],
                 .step = step,
                 .out_params = updated_params_buffer,
                 .out_params_liveness = updated_liveness_buffer,
                 .tips = tips_buffer,
                 .tip_lengths = tip_length_buffer,
                 .n_tracks_per_seed = n_tracks_per_seed_buffer,
                 .tmp_params = tmp_params_buffer,
                 .tmp_links = tmp_links_buffer,
                 .jacobian = jacobian_buffer,
                 .tmp_jacobian = tmp_jacobian_buffer,
                 .link_predicted_parameter = link_predicted_parameter_buffer,
                 .link_filtered_parameter = link_filtered_parameter_buffer});

            std::swap(in_params_buffer, updated_params_buffer);
            std::swap(param_liveness_buffer, updated_liveness_buffer);

            if (mr().host) {
                vecmem::async_size size =
                    copy().get_size(links_buffer, *(mr().host));
                // Here we could give control back to the caller, once our code
                // allows for it. (coroutines...)
                step_to_link_idx_map[step + 1] = size.get();
            } else {
                step_to_link_idx_map[step + 1] = copy().get_size(links_buffer);
            }
            n_candidates =
                step_to_link_idx_map[step + 1] - step_to_link_idx_map[step];
        }

        // Set up the buffer used in the next few steps
        vecmem::data::vector_buffer<unsigned int> param_ids_buffer(n_candidates,
                                                                   mr().main);
        copy().setup(param_ids_buffer)->ignore();

        /*
         * On later steps, we can duplicate removal which will attempt to find
         * tracks that are propagated multiple times and deduplicate them.
         */
        if ((n_candidates > 0) &&
            (step >= m_data->m_config.duplicate_removal_minimum_length)) {

            vecmem::data::vector_buffer<unsigned int>
                link_last_measurement_buffer(n_candidates, mr().main);
            copy().setup(link_last_measurement_buffer)->ignore();

            /*
             * First, we sort the tracks by the index of their final
             * measurement which is critical to ensure good performance.
             */
            fill_finding_duplicate_removal_sort_keys_kernel(
                n_candidates,
                {.links_view = links_buffer,
                 .param_liveness_view = param_liveness_buffer,
                 .link_last_measurement_view = link_last_measurement_buffer,
                 .param_ids_view = param_ids_buffer,
                 .n_links = n_candidates,
                 .curr_links_idx = step_to_link_idx_map[step],
                 .n_measurements = n_measurements});

            sort_param_ids_by_last_measurement(link_last_measurement_buffer,
                                               param_ids_buffer);

            /*
             * Then, we run the actual duplicate removal kernel.
             */
            remove_duplicates_kernel(
                n_candidates, m_data->m_config,
                {.links_view = links_buffer,
                 .link_last_measurement_view = link_last_measurement_buffer,
                 .param_ids_view = param_ids_buffer,
                 .param_liveness_view = param_liveness_buffer,
                 .n_links = n_candidates,
                 .curr_links_idx = step_to_link_idx_map[step],
                 .n_measurements = n_measurements,
                 .step = step});
        }

        // If no more CKF step is expected, the tips and links are populated,
        // and any further time-consuming action is avoided
        if (step == (m_data->m_config.max_track_candidates_per_track - 1)) {
            break;
        }

        if (n_candidates > 0) {

            /*****************************************************************
             * Get key and value for parameter sorting
             *****************************************************************/
            {
                vecmem::data::vector_buffer<device::sort_key> keys_buffer(
                    n_candidates, mr().main);
                copy().setup(keys_buffer)->ignore();

                fill_finding_propagation_sort_keys_kernel(
                    n_candidates, {.params_view = in_params_buffer,
                                   .param_liveness_view = param_liveness_buffer,
                                   .keys_view = keys_buffer,
                                   .ids_view = param_ids_buffer});

                // Sort the key and values
                sort_param_ids_by_keys(keys_buffer, param_ids_buffer);
            }

            /*****************************************************************
             * Propagate to the next surface
             *****************************************************************/
            if (m_data->m_config.run_mbf_smoother) {
                tmp_jacobian_buffer = {n_candidates, mr().main};
            }

            propagate_to_next_surface_kernel(
                n_candidates, {.config = m_data->m_config,
                               .det = det,
                               .field = bfield,
                               .params = in_params_buffer,
                               .params_liveness = param_liveness_buffer,
                               .param_ids = param_ids_buffer,
                               .links = links_buffer,
                               .prev_links_idx = step_to_link_idx_map[step],
                               .step = step,
                               .tips = tips_buffer,
                               .tip_lengths = tip_length_buffer,
                               .tmp_jacobian = tmp_jacobian_buffer});
        }

        n_in_params = n_candidates;
    }

    tmp_jacobian_buffer = {};

    TRACCC_DEBUG(
        "Final link buffer usage was "
        << copy().get_size(links_buffer) << " out of " << link_buffer_capacity
        << " ("
        << ((100.f * static_cast<float>(copy().get_size(links_buffer))) /
            static_cast<float>(link_buffer_capacity))
        << "%)");

    /*****************************************************************
     * Kernel6: Build tracks
     *****************************************************************/

    // Get the number of tips
    unsigned int n_tips_total = 0u;
    if (mr().host) {
        vecmem::async_size size = copy().get_size(tips_buffer, *(mr().host));
        // Here we could give control back to the caller, once our code allows
        // for it. (coroutines...)
        n_tips_total = size.get();
    } else {
        n_tips_total = copy().get_size(tips_buffer);
    }

    vecmem::data::vector_buffer<unsigned int> tip_to_output_map;

    if (n_tips_total > 0 &&
        m_data->m_config.max_num_tracks_per_measurement > 0) {
        // TODO: DOCS

        vecmem::data::vector_buffer<unsigned int>
            best_tips_per_measurement_index_buffer(
                m_data->m_config.max_num_tracks_per_measurement *
                    n_measurements,
                mr().main);
        copy().setup(best_tips_per_measurement_index_buffer)->ignore();

        vecmem::data::vector_buffer<unsigned long long int>
            best_tips_per_measurement_insertion_mutex_buffer(n_measurements,
                                                             mr().main);
        copy()
            .setup(best_tips_per_measurement_insertion_mutex_buffer)
            ->ignore();
        copy()
            .memset(best_tips_per_measurement_insertion_mutex_buffer, 0)
            ->ignore();

        {
            vecmem::data::vector_buffer<scalar>
                best_tips_per_measurement_pval_buffer(
                    m_data->m_config.max_num_tracks_per_measurement *
                        n_measurements,
                    mr().main);
            copy().setup(best_tips_per_measurement_pval_buffer)->ignore();

            gather_best_tips_per_measurement_kernel(
                n_tips_total,
                {tips_buffer, links_buffer, measurements_view,
                 best_tips_per_measurement_insertion_mutex_buffer,
                 best_tips_per_measurement_index_buffer,
                 best_tips_per_measurement_pval_buffer,
                 m_data->m_config.max_num_tracks_per_measurement});
        }

        vecmem::data::vector_buffer<unsigned int> votes_per_tip_buffer(
            n_tips_total, mr().main);
        copy().setup(votes_per_tip_buffer)->ignore();
        copy().memset(votes_per_tip_buffer, 0)->ignore();

        gather_measurement_votes_kernel(
            m_data->m_config.max_num_tracks_per_measurement * n_measurements,
            {.insertion_mutex =
                 best_tips_per_measurement_insertion_mutex_buffer,
             .tip_index = best_tips_per_measurement_index_buffer,
             .votes_per_tip = votes_per_tip_buffer,
             .max_num_tracks_per_measurement =
                 m_data->m_config.max_num_tracks_per_measurement});

        tip_to_output_map =
            vecmem::data::vector_buffer<unsigned int>(n_tips_total, mr().main);
        copy().setup(tip_to_output_map)->ignore();

        vecmem::data::vector_buffer<unsigned int> new_tip_length_buffer{
            n_tips_total, mr().main, vecmem::data::buffer_type::resizable};
        copy().setup(new_tip_length_buffer)->ignore();

        update_tip_length_buffer_kernel(
            n_tips_total,
            {.old_tip_length = tip_length_buffer,
             .new_tip_length = new_tip_length_buffer,
             .measurement_votes = votes_per_tip_buffer,
             .tip_to_output_map = tip_to_output_map,
             .min_measurement_voting_fraction =
                 m_data->m_config.min_measurement_voting_fraction});

        tip_length_buffer = std::move(new_tip_length_buffer);
    }

    vecmem::vector<unsigned int> tips_length_host(mr().host);
    vecmem::copy::event_type ev = copy()(tip_length_buffer, tips_length_host);
    // Here we could give control back to the caller, once our code allows for
    // it. (coroutines...)
    ev->wait();
    // The following is only necessary if filtering was not turned on. Since
    // with filtering on, the buffer is resizable, so the host vector would
    // already have the correct size.
    if (m_data->m_config.max_num_tracks_per_measurement == 0) {
        tips_length_host.resize(n_tips_total);
    }

    unsigned int n_states = 0u;
    if (m_data->m_config.run_mbf_smoother) {
        n_states = std::accumulate(tips_length_host.begin(),
                                   tips_length_host.end(), 0u);
    }

    // Create track buffer
    typename edm::track_container<default_algebra>::buffer
        track_candidates_buffer{
            {tips_length_host, mr().main, mr().host},
            {n_states, mr().main, vecmem::data::buffer_type::resizable},
            measurements_view};
    copy().setup(track_candidates_buffer.tracks)->ignore();
    copy().setup(track_candidates_buffer.states)->ignore();

    // @Note: nBlocks can be zero in case there is no tip. This happens when
    // chi2_max config is set tightly and no tips are found
    if (n_tips_total > 0) {
        build_tracks_kernel(
            n_tips_total, m_data->m_config.run_mbf_smoother,
            {.seeds_view = seeds,
             .links_view = links_buffer,
             .tips_view = tips_buffer,
             .tracks_view = {track_candidates_buffer},
             .tip_to_output_map = tip_to_output_map,
             .jacobian_ptr = jacobian_buffer.ptr(),
             .link_predicted_parameter_view = link_predicted_parameter_buffer,
             .link_filtered_parameter_view = link_filtered_parameter_buffer});
    }

    return track_candidates_buffer;
}

}  // namespace traccc::device
