/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::edm {

template <typename BASE>
TRACCC_HOST_DEVICE void track_fit<BASE>::reset_quality() {

    ndf() = {};
    chi2() = {};
    pval() = {};
    nholes() = {};
}

}  // namespace traccc::edm
