/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/device/concepts/barrier.hpp"

// SYCL include(s).
#include <sycl/sycl.hpp>

namespace traccc::sycl::details {

/// A barrier type for SYCL kernels.
template <std::size_t DIMENSIONS = 1>
struct barrier {

    /// Verify that the barrier is used on a supported dimensionality.
    static_assert(DIMENSIONS > 0 && DIMENSIONS < 4,
                  "Only 1D, 2D, and 3D barriers are supported.");

    /// Dimensions of the barrier
    static constexpr std::size_t dimensions = DIMENSIONS;

    /// Constructor
    ///
    /// @param item The @c ::sycl::nd_item<1> to use for the barrier.
    ///
    explicit barrier(const ::sycl::nd_item<dimensions>& item) : m_item(item) {}

    inline void blockBarrier() const {
        ::sycl::group_barrier(m_item.get_group());
    }

    inline bool blockAnd(bool predicate) const {
        blockBarrier();
        return ::sycl::all_of_group(m_item.get_group(), predicate);
    }

    inline bool blockOr(bool predicate) const {
        blockBarrier();
        return ::sycl::any_of_group(m_item.get_group(), predicate);
    }

    inline unsigned int blockCount(bool predicate) const {
        blockBarrier();
        return ::sycl::reduce_over_group(m_item.get_group(),
                                         predicate ? 1u : 0u, ::sycl::plus<>());
    }

    private:
    /// Item object coming from the SYCL kernel
    const ::sycl::nd_item<dimensions>& m_item;
};

/// Verify that @c traccc::sycl::details::barrier fulfills the
/// @c traccc::device::concepts::barrier concept.
static_assert(traccc::device::concepts::barrier<barrier<1>>);
static_assert(traccc::device::concepts::barrier<barrier<2>>);
static_assert(traccc::device::concepts::barrier<barrier<3>>);

}  // namespace traccc::sycl::details
