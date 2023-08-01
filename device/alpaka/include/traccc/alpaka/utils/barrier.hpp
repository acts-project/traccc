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

    ALPAKA_FN_INLINE ALPAKA_FN_ACC barrier(const TAcc* acc) : m_acc(acc) {};

    ALPAKA_FN_ACC
    void blockBarrier() { ::alpaka::syncBlockThreads(*m_acc); }

    ALPAKA_FN_ACC
    bool blockOr(bool predicate) { return ::alpaka::syncBlockThreadsPredicate<::alpaka::BlockCount>(*m_acc, predicate); }

    private:
    const TAcc* m_acc;
};

}  // namespace traccc::alpaka
