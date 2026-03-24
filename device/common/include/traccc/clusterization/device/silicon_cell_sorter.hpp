/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"

namespace traccc::device {

/// Functor to help with sorting cells on a device
class silicon_cell_sorter {

    public:
    /// Constructor, capturing the silicon cells to sort indices by
    silicon_cell_sorter(const edm::silicon_cell_collection::const_view& cells)
        : m_cells(cells) {}

    /// Index comparison operator
    ///
    /// The logic is a bit convoluted here. :-( The index sequence/vector that
    /// we sort may be larger than the cell collection. (Whenever we are dealing
    /// with resizable cell collections. Which may happen sometimes.) In this
    /// case the behaviour should be that for the non-existent cells the indices
    /// should be left unchanged. Which the following logic should do...
    ///
    TRACCC_HOST_DEVICE bool operator()(unsigned int lhs,
                                       unsigned int rhs) const {

        const edm::silicon_cell_collection::const_device cells{m_cells};
        if (lhs >= cells.size()) {
            return false;
        }
        if (rhs >= cells.size()) {
            return true;
        }
        return cells.at(lhs) < cells.at(rhs);
    }

    private:
    /// The cells to sort indices by
    edm::silicon_cell_collection::const_view m_cells;

};  // class silicon_cell_sorter

}  // namespace traccc::device
