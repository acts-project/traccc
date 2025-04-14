/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"

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

    // Create resolved candidate buffer
    track_candidate_container_types::buffer res_candidates_buffer{
        {10, m_mr.main},
        {std::vector<std::size_t>(10, 10), m_mr.main, m_mr.host,
         vecmem::data::buffer_type::resizable}};

    return res_candidates_buffer;
}

}  // namespace traccc::cuda