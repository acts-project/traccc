/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/qualifiers.hpp"

namespace traccc::sycl {
struct thread_id1 {
    TRACCC_DEVICE thread_id1(const ::sycl::nd_item<1>& item) : m_item(item) {}

    auto inline TRACCC_DEVICE getLocalThreadId() const {
        return m_item.get_local_linear_id();
    }

    auto inline TRACCC_DEVICE getLocalThreadIdX() const {
        return m_item.get_local_linear_id();
    }

    auto inline TRACCC_DEVICE getGlobalThreadId() const {
        return m_item.get_global_linear_id();
    }

    auto inline TRACCC_DEVICE getGlobalThreadIdX() const {
        return m_item.get_global_linear_id();
    }

    auto inline TRACCC_DEVICE getBlockIdX() const {
        return m_item.get_group_linear_id();
    }

    auto inline TRACCC_DEVICE getBlockDimX() const {
        return m_item.get_local_range(0);
    }

    auto inline TRACCC_DEVICE getGridDimX() const {
        return m_item.get_global_range(0);
    }

    private:
    const ::sycl::nd_item<1>& m_item;
};
}  // namespace traccc::sycl
