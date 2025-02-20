/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/pair.hpp"

namespace traccc {

// A link that contains the index of corresponding measurement and the index of
// a link from a previous step of track finding
struct candidate_link {

    // Type of index
    using link_index_type = traccc::pair<unsigned int, unsigned int>;

    // Index of link from the previous step
    link_index_type previous;

    // Measurement index
    unsigned int meas_idx;

    // Index to the initial seed
    unsigned int seed_idx;

    // How many times it skipped a surface
    unsigned int n_skipped;

    // chi2
    traccc::scalar chi2;
};

}  // namespace traccc
