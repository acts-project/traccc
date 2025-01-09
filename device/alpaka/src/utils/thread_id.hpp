/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "utils.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"

// Alpaka include(s).
#include <alpaka/alpaka.hpp>

namespace traccc::alpaka::details {

/// An Alpaka thread identifier type
template <typename Acc>
struct thread_id1 {
    TRACCC_HOST_DEVICE explicit thread_id1(const Acc& acc) : m_acc(acc) {}

    unsigned int inline TRACCC_HOST_DEVICE getLocalThreadId() const {
        return static_cast<unsigned int>(
            ::alpaka::getIdx<::alpaka::Block, ::alpaka::Threads>(m_acc)[0u]);
    }

    unsigned int inline TRACCC_HOST_DEVICE getLocalThreadIdX() const {
        return getLocalThreadId();
    }

    unsigned int inline TRACCC_HOST_DEVICE getGlobalThreadId() const {
        return getLocalThreadId() + getBlockIdX() * getBlockDimX();
    }

    unsigned int inline TRACCC_HOST_DEVICE getGlobalThreadIdX() const {
        return getLocalThreadId() + getBlockIdX() * getBlockDimX();
    }

    unsigned int inline TRACCC_HOST_DEVICE getBlockIdX() const {
        return static_cast<unsigned int>(
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Blocks>(m_acc)[0u]);
    }

    unsigned int inline TRACCC_HOST_DEVICE getBlockDimX() const {
        return static_cast<unsigned int>(
            ::alpaka::getWorkDiv<::alpaka::Block, ::alpaka::Threads>(
                m_acc)[0u]);
    }

    unsigned int inline TRACCC_HOST_DEVICE getGridDimX() const {
        return static_cast<unsigned int>(
            ::alpaka::getWorkDiv<::alpaka::Grid, ::alpaka::Blocks>(m_acc)[0u]);
    }

    private:
    const Acc& m_acc;
};

/// Verify that @c traccc::alpaka::details::thread_id1 fulfills the
/// @c traccc::device::concepts::thread_id1 concept.
static_assert(traccc::device::concepts::thread_id1<thread_id1<Acc>>);

}  // namespace traccc::alpaka::details
