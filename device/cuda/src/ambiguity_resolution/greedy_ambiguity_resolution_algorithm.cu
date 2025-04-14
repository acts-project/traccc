/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"

namespace traccc::cuda {

greedy_ambiguity_resolution_algorithm::greedy_ambiguity_resolution_algorithm(
    const config_type& cfg, const traccc::memory_resource& mr,
    vecmem::copy& copy, stream& str, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_config(config),
      m_mr(mr),
      m_copy(copy),
      m_stream(str)
}

greedy_ambiguity_resolution_algorithm::output_type
greedy_ambiguity_resolution_algorithm_algorithm::operator()(
    const track_candidate_container_types::const_view& track_candidates_view)
    const {



}

}  // namespace traccc::cuda