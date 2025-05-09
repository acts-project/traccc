/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

// Update result from the iterations in ambiguity resolver
struct update_result {
    unsigned int max_shared;
    unsigned int n_updated_tracks;
    unsigned int n_accepted;
};

}  // namespace traccc::device