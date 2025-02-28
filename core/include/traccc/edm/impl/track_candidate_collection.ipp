/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::edm {

template <typename BASE>
template <typename T>
TRACCC_HOST_DEVICE bool track_candidate<BASE>::operator==(
    const track_candidate<T>& other) const {

    return ((params() == other.params()) && (ndf() == other.ndf()) &&
            (chi2() == other.chi2()) && (pval() == other.pval()) &&
            (nholes() == other.nholes()) &&
            (measurement_indices() == other.measurement_indices()));
}

}  // namespace traccc::edm
