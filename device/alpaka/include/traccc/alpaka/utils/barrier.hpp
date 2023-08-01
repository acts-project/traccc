/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

namespace traccc::alpaka {

template <typename TAcc>
struct barrier {

    barrier(TAcc const& acc) : m_acc(acc) {};

    // TRACCC_DEVICE
    // void blockBarrier() { ::alpaka::syncBlockThreads(m_acc); }

    // TRACCC_DEVICE
    // bool blockOr(bool predicate) { return ::alpaka::syncBlockThreadsPredicate<::alpaka::BlockCount>(m_acc, predicate); }

    TRACCC_DEVICE
    void blockBarrier() { return; }

    TRACCC_DEVICE
    bool blockOr(bool predicate) { return predicate; }

    private:
    TAcc m_acc;
};

}  // namespace traccc::alpaka
