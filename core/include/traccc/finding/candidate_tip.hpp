/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc {

// A track candidate tip, pointing to the last candidate on a potential track.
// Models the beginning of a reversed singularly linked list.
struct candidate_tip {
    using index_t = unsigned int;

    /// The step index of the candidate pointed at.
    index_t step_idx;

    /// The candidate index in the list belonging to `step_idx`.
    index_t candidate_idx;
};

}  // namespace traccc
