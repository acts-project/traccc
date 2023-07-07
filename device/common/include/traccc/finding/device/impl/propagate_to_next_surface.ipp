/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

template <typename propagator_t, typename config_t>
TRACCC_DEVICE inline void propagate_to_next_surface(
    std::size_t globalIndex, const config_t cfg,
    typename propagator_t::detector_type::detector_view_type det_data,
    vecmem::data::jagged_vector_view<typename propagator_t::intersection_type>
        nav_candidates_buffer,
    bound_track_parameters_collection_types::const_view in_params_view,
    vecmem::data::vector_view<const candidate_link> links_view,
    const unsigned int step, const unsigned int& n_in_params,
    bound_track_parameters_collection_types::view out_params_view,
    vecmem::data::vector_view<unsigned int> param_to_link_view,
    vecmem::data::vector_view<typename candidate_link::link_index_type>
        tips_view,
    unsigned int& n_out_params) {

    if (globalIndex >= n_in_params) {
        return;
    }

    // Detector
    typename propagator_t::detector_type det(det_data);

    // Navigation candidate buffer
    vecmem::jagged_device_vector<typename propagator_t::intersection_type>
        nav_candidates(nav_candidates_buffer);

    // Input parameters
    bound_track_parameters_collection_types::const_device in_params(
        in_params_view);

    // Links
    vecmem::device_vector<const candidate_link> links(links_view);

    // Out parameters
    bound_track_parameters_collection_types::device out_params(out_params_view);

    // Param to Link ID
    vecmem::device_vector<unsigned int> param_to_link(param_to_link_view);

    // tips
    vecmem::device_vector<typename candidate_link::link_index_type> tips(
        tips_view);

    // Input bound track parameter
    const bound_track_parameters in_par = in_params.at(globalIndex);

    // Create propagator
    propagator_t propagator({}, {});

    // Create propagator state
    typename propagator_t::state propagation(
        in_par, det.get_bfield(), det,
        std::move(nav_candidates.at(globalIndex)));
    propagation._stepping
        .template set_constraint<detray::step::constraint::e_accuracy>(
            cfg.constrained_step_size);

    // Actor state
    // @TODO: simplify the syntax here
    // @NOTE: Post material interaction might be required here
    using actor_list_type =
        typename propagator_t::actor_chain_type::actor_list_type;
    typename detray::detail::tuple_element<0, actor_list_type>::type::state
        s0{};
    typename detray::detail::tuple_element<1, actor_list_type>::type::state
        s1{};
    typename detray::detail::tuple_element<3, actor_list_type>::type::state
        s3{};
    typename detray::detail::tuple_element<2, actor_list_type>::type::state s2{
        s3};
    typename detray::detail::tuple_element<4, actor_list_type>::type::state s4{
        cfg.min_step_length_for_surface_aborter};

    // @TODO: Should be removed once detray is fixed to set the volume in the
    // constructor
    propagation._navigation.set_volume(in_par.surface_link().volume());

    // Propagate to the next surface
    propagator.propagate_sync(propagation, std::tie(s0, s1, s2, s3, s4));

    // If a surface found, add the parameter for the next step
    if (s4.success) {
        vecmem::device_atomic_ref<unsigned int> num_out_params(n_out_params);
        const unsigned int out_param_id = num_out_params.fetch_add(1);

        out_params[out_param_id] = propagation._stepping._bound_params;
        param_to_link[out_param_id] = globalIndex;
    }
    // Unless the track found a surface, it is considered a tip
    else if (!s4.success && step >= cfg.min_track_candidates_per_track - 1) {
        tips.push_back({step, globalIndex});
    }
}

}  // namespace traccc::device