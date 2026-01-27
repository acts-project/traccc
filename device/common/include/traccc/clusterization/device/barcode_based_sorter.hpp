/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

// Detray include(s).
#include <detray/geometry/barcode.hpp>

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

/// Functor to help with sorting measurements based on their surface barcodes
class barcode_based_sorter {

    public:
    /// Constructor, capturing the barcodes to sort indices by
    barcode_based_sorter(
        const vecmem::device_vector<const detray::geometry::barcode>& barcodes)
        : m_barcodes(barcodes) {}

    /// Index comparison operator
    ///
    /// The logic is a bit convoluted here. :-( The index sequence/vector that
    /// we sort may be larger than the barcode vector. (Whenever we are dealing)
    /// with resizable measurement collections. Which is often.) In this case
    /// the behaviour should be that for the non-existent barcodes the indices
    /// should be left unchanged. Which the following logic should do...
    ///
    TRACCC_HOST_DEVICE bool operator()(unsigned int lhs,
                                       unsigned int rhs) const {

        if (lhs >= m_barcodes.size()) {
            return false;
        }
        if (rhs >= m_barcodes.size()) {
            return true;
        }
        return m_barcodes.at(lhs) < m_barcodes.at(rhs);
    }

    private:
    /// The barcodes to sort indices by
    vecmem::device_vector<const detray::geometry::barcode> m_barcodes;

};  // class barcode_based_sorter

}  // namespace traccc::device
