/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/detectors/bfield.hpp"
#include "detray/detectors/toy_metadata.hpp"

// Detray test include(s)
#include "detray/test/device/propagator_test.hpp"

// Vecmem include(s)
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// Covfie include(s)
#include <covfie/cuda/backend/primitive/cuda_device_array.hpp>

namespace detray {

namespace bfield::cuda {

// Inhomogeneous field (cuda)
using inhom_bknd_t = covfie::backend::affine<covfie::backend::linear<
    covfie::backend::strided<covfie::vector::vector_d<std::size_t, 3>,
                             covfie::backend::cuda_device_array<
                                 covfie::vector::vector_d<scalar, 3>>>>>;

}  // namespace bfield::cuda

/// Launch the propagation test kernel
template <typename bfield_bknd_t, typename detector_t>
void propagator_test(
    typename detector_t::view_type, const propagation::config &,
    covfie::field_view<bfield_bknd_t>, vecmem::data::vector_view<track_t> &,
    vecmem::data::jagged_vector_view<detail::step_data<algebra_t>> &);

/// Test function for propagator on the device
template <typename bfield_bknd_t, typename detector_t>
inline auto run_propagation_device(
    vecmem::memory_resource *mr, const propagation::config &cfg,
    typename detector_t::view_type det_view,
    covfie::field_view<bfield_bknd_t> field_data, dvector<track_t> &tracks,
    const vecmem::jagged_vector<detail::step_data<algebra_t>> &host_steps)
    -> vecmem::jagged_vector<detail::step_data<algebra_t>> {

    // Helper object for performing memory copies.
    vecmem::copy copy;

    // Get tracks data
    auto tracks_data = vecmem::get_data(tracks);

    // Create vector buffer for track recording
    std::vector<std::size_t> sizes(tracks.size(), 0);
    std::vector<std::size_t> capacities;
    for (auto &st : host_steps) {
        // Add a few more elements for security (in case the device finds more
        // surfaces)
        capacities.push_back(st.size() + 10u);
    }

    vecmem::data::jagged_vector_buffer<detail::step_data<algebra_t>>
        steps_buffer(capacities, *mr, nullptr,
                     vecmem::data::buffer_type::resizable);

    copy.setup(steps_buffer)->wait();

    // Run the propagator test for GPU device
    propagator_test<bfield_bknd_t, detector_t>(det_view, cfg, field_data,
                                               tracks_data, steps_buffer);

    vecmem::jagged_vector<detail::step_data<algebra_t>> steps(mr);

    copy(steps_buffer, steps)->wait();

    return steps;
}

/// Test chain for the propagator
template <typename device_bfield_bknd_t, typename host_bfield_bknd_t,
          typename detector_t>
inline auto run_propagation_test(vecmem::memory_resource *mr, detector_t &det,
                                 const propagator_test_config &cfg,
                                 typename detector_t::view_type det_view,
                                 covfie::field<host_bfield_bknd_t> &&field) {

    // Create the vector of initial track parameterizations
    auto tracks_host = generate_tracks<generator_t>(mr, cfg.track_generator);
    vecmem::vector<track_t> tracks_device(tracks_host, mr);

    // Host propagation
    auto host_steps =
        run_propagation_host(mr, det, cfg.propagation, field, tracks_host);

    // Device propagation (device backend specific implementation)
    covfie::field<device_bfield_bknd_t> device_field(field);
    auto device_steps =
        run_propagation_device<device_bfield_bknd_t, detector_t>(
            mr, cfg.propagation, det_view, device_field, tracks_device,
            host_steps);

    // Check the results
    compare_propagation_results(host_steps, device_steps);
}

}  // namespace detray
