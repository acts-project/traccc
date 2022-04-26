/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"

namespace traccc {

/// Implemementation of SparseCCL, following
/// [DOI: 10.1109/DASIP48288.2019.9049184]
///
/// Requires cells to be sorted in column major
namespace detail {

/// Find root of the tree for entry @param e
///
/// @param L an equivalance table
///
/// @return the root of @param e
template <template <typename> class vector_t>
TRACCC_HOST_DEVICE inline unsigned int find_root(
    const vector_t<unsigned int>& L, const unsigned int& e) {
    unsigned int r = e;
    while (L[r] != r) {
        r = L[r];
    }
    return r;
}

/// Create a union of two entries @param e1 and @param e2
///
/// @param L an equivalance table
///
/// @return the rleast common ancestor of the entries
template <template <typename> class vector_t>
TRACCC_HOST_DEVICE inline unsigned int make_union(vector_t<unsigned int>& L,
                                                  const unsigned int& e1,
                                                  const unsigned int& e2) {
    int e;
    if (e1 < e2) {
        e = e1;
        L[e2] = e;
    } else {
        e = e2;
        L[e1] = e;
    }
    return e;
}

/// Helper method to find adjacent cells
///
/// @param a the first cell
/// @param b the second cell
///
/// @return boolan to indicate 8-cell connectivity
template <typename cell_t>
TRACCC_HOST_DEVICE inline bool is_adjacent(cell_t a, cell_t b) {
    return (a.channel0 - b.channel0) * (a.channel0 - b.channel0) <= 1 and
           (a.channel1 - b.channel1) * (a.channel1 - b.channel1) <= 1;
}

/// Helper method to find define distance,
/// does not need abs, as channels are sorted in
/// column major
///
/// @param a the first cell
/// @param b the second cell
///
/// @return boolan to indicate !8-cell connectivity
template <typename cell_t>
TRACCC_HOST_DEVICE inline bool is_far_enough(cell_t a, cell_t b) {
    return (a.channel1 - b.channel1) > 1;
}

/// Sparce CCL algorithm
///
template <template <typename> class vector_t, typename cell_t>
TRACCC_HOST_DEVICE inline void sparse_ccl(const vector_t<cell_t>& cells,
                                          vector_t<unsigned int>& L,
                                          unsigned int& labels) {

    // Internal list linking

    // first scan: pixel association
    unsigned int start_j = 0;
    for (unsigned int i = 0; i < cells.size(); ++i) {
        L[i] = i;
        int ai = i;
        if (i > 0) {
            for (unsigned int j = start_j; j < i; ++j) {
                if (is_adjacent<cell_t>(cells[i], cells[j])) {
                    ai = make_union<vector_t>(L, ai, find_root<vector_t>(L, j));
                } else if (is_far_enough<cell_t>(cells[i], cells[j])) {
                    ++start_j;
                }
            }
        }
    }

    // second scan: transitive closure
    for (unsigned int i = 0; i < cells.size(); ++i) {
        unsigned int l = 0;
        if (L[i] == i) {
            ++labels;
            l = labels;
        } else {
            l = L[L[i]];
        }
        L[i] = l;
    }
}
}  // namespace detail

}  // namespace traccc
