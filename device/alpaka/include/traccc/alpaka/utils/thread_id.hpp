/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include "traccc/definitions/qualifiers.hpp"

namespace traccc::alpaka {
template <typename Acc>
struct thread_id1 {
    TRACCC_DEVICE thread_id1(const Acc& acc) : m_acc(acc) {}

    auto inline TRACCC_DEVICE getLocalThreadId() const {
        return ::alpaka::getIdx<::alpaka::Block, ::alpaka::Threads>(m_acc)[0u];
    }

    auto inline TRACCC_DEVICE getLocalThreadIdX() const {
        return getLocalThreadId();
    }

    auto inline TRACCC_DEVICE getGlobalThreadId() const {
        return getLocalThreadId() + getBlockIdX() * getBlockDimX();
    }

    auto inline TRACCC_DEVICE getGlobalThreadIdX() const {
        return getLocalThreadId() + getBlockIdX() * getBlockDimX();
    }

    auto inline TRACCC_DEVICE getBlockIdX() const {
        return ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Blocks>(m_acc)[0u];
    }

    auto inline TRACCC_DEVICE getBlockDimX() const {
        return ::alpaka::getWorkDiv<::alpaka::Block, ::alpaka::Threads>(
            m_acc)[0u];
    }

    auto inline TRACCC_DEVICE getGridDimX() const {
        return ::alpaka::getWorkDiv<::alpaka::Grid, ::alpaka::Blocks>(
            m_acc)[0u];
    }

    private:
    const Acc& m_acc;
};
}  // namespace traccc::alpaka
