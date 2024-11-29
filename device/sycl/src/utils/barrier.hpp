/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

// SYCL includes
#include <sycl/sycl.hpp>

namespace traccc::sycl {

struct barrier {
    barrier(::sycl::nd_item<1> item) : m_item(item) {}

    TRACCC_DEVICE
    void blockBarrier() { m_item.barrier(); }

    TRACCC_DEVICE
    bool blockAnd(bool predicate) {
        m_item.barrier();
        return ::sycl::all_of_group(m_item.get_group(), predicate);
    }

    TRACCC_DEVICE
    bool blockOr(bool predicate) {
        m_item.barrier();
        return ::sycl::any_of_group(m_item.get_group(), predicate);
    }

    TRACCC_DEVICE
    unsigned int blockCount(bool predicate) {
        m_item.barrier();
        return ::sycl::reduce_over_group(m_item.get_group(),
                                         predicate ? 1u : 0u, ::sycl::plus<>());
    }

    private:
    ::sycl::nd_item<1> m_item;
};

}  // namespace traccc::sycl
