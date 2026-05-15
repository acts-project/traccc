/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cassert>

namespace traccc::device {

template <typename detector_list_t, typename bfield_list_t>
auto kalman_fitting_algorithm::prepare_fit_payload_helper(
    const detector_buffer& det, const magnetic_field& field,
    const std::vector<unsigned int>& n_surfaces,
    const vecmem::data::vector_view<const unsigned int>& track_indices,
    vecmem::data::vector_view<unsigned int>& track_liveness,
    edm::track_container<default_algebra>::view tracks) const
    -> std::unique_ptr<fit_payload_base> {

    return detector_buffer_magnetic_field_visitor<detector_list_t,
                                                  bfield_list_t>(
        det, field,
        [&]<typename detector_traits_t, typename bfield_view_t>(
            const typename detector_traits_t::view& detector,
            const bfield_view_t& bfield) -> std::unique_ptr<fit_payload_base> {
            // Type of the concrete payload to create.
            using payload_type =
                fit_payload<typename detector_traits_t::device, bfield_view_t>;

            // Create the surface buffer for the payload object.
            vecmem::data::jagged_vector_buffer<
                typename detector_traits_t::device::surface_type>
                surfaces{n_surfaces, mr().main, mr().host,
                         vecmem::data::buffer_type::resizable};
            copy().setup(surfaces)->ignore();

            // Create a fit payload object.
            typename payload_type::payload_type host_payload{
                .det_data = detector,
                .field_data = bfield,
                .param_ids_view = track_indices,
                .param_liveness_view = track_liveness,
                .tracks_view = tracks,
                .surfaces_view = surfaces};

            // Now create the "polymorphic" fit payload object.
            auto payload =
                std::make_unique<payload_type>(det, field, host_payload);

            // Move the surface buffer into it.
            payload->surfaces = std::move(surfaces);

            // Now set up the device payload buffer, and copy the host payload
            // into it.
            payload->device_payload = {1u, mr().main};
            copy().setup(payload->device_payload)->ignore();
            copy()(
                vecmem::data::vector_view<const typename fit_payload<
                    typename detector_traits_t::device,
                    bfield_view_t>::payload_type>(1u, &(payload->host_payload)),
                payload->device_payload)
                ->ignore();

            // All done, we can return the created payload.
            return payload;
        });
}

template <typename detector_t, typename bfield_t>
auto kalman_fitting_algorithm::cast_fit_payload(const fit_payload_base& payload)
    -> const fit_payload<detector_t, bfield_t>& {

    // Final type to cast to.
    using result_type = fit_payload<detector_t, bfield_t>;

    // Check that the payload buffer can be converted to the requested type. But
    // only do so during debugging.
    assert(dynamic_cast<const result_type*>(&payload) != nullptr);

    // In optimised builds just do the cast without the check.
    return static_cast<const result_type&>(payload);
}

}  // namespace traccc::device
