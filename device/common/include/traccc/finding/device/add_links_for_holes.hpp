/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

namespace traccc::device {

/// Function to add a dummy link in case of a hole

TRACCC_DEVICE inline void add_links_for_holes(
    std::size_t globalIndex,
    vecmem::data::vector_view<const unsigned int> n_candidates_view,
    bound_track_parameters_collection_types::const_view in_params_view,
    vecmem::data::vector_view<const candidate_link> prev_links_view,
    vecmem::data::vector_view<const unsigned int> prev_param_to_link_view,
    const unsigned int step, const unsigned int& n_max_candidates,
    bound_track_parameters_collection_types::view out_params_view,
    vecmem::data::vector_view<candidate_link> links_view,
    unsigned int& n_total_candidates);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/finding/device/impl/add_links_for_holes.ipp"