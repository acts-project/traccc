/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../utils/utils.hpp"
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

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

    vecmem::data::vector_buffer<unsigned int> accepted_ids_buffer{n_tracks,
                                                                  m_mr.main};
    vecmem::device_vector<unsigned int> accepted_ids_device(
        accepted_ids_buffer);
    thrust::sequence(thrust_policy, accepted_ids_device.begin(),
                     accepted_ids_device.end(), 0, 1);

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

    // Make measurement ID, chi2 and n_measurement vector
    vecmem::data::jagged_vector_buffer<detray::geometry::barcode> meas_ids{
        meas_sizes, m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable};

    vecmem::data::vector_buffer<traccc::scalar> chi_squares_buffer(n_tracks,
                                                                   m_mr.main);
    vecmem::data::vector_buffer<std::size_t> n_meas(n_tracks, m_mr.main);

    // Create resolved candidate buffer
    track_candidate_container_types::buffer res_candidates_buffer{
        {10, m_mr.main},
        {std::vector<std::size_t>(10, 10), m_mr.main, m_mr.host,
         vecmem::data::buffer_type::resizable}};

    return res_candidates_buffer;
}

}  // namespace traccc::cuda