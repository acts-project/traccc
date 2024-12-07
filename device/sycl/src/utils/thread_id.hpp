/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/device/concepts/thread_id.hpp"

// SYCL include(s).
#include <sycl/sycl.hpp>

namespace traccc::sycl::details {

/// A SYCL thread identifier type
template <std::size_t DIMENSIONS = 1>
struct thread_id {

    /// Verify that the thread identifier is used on a supported dimensionality.
    static_assert(DIMENSIONS > 0 && DIMENSIONS < 4,
                  "Only 1D, 2D, and 3D barriers are supported.");

    /// Dimensions of the thread identifier
    static constexpr std::size_t dimensions = DIMENSIONS;

    /// Constructor
    ///
    /// @param item The @c ::sycl::nd_item<1> to use for the thread ID.
    ///
    explicit thread_id(const ::sycl::nd_item<dimensions>& item)
        : m_item(item) {}

    inline auto getLocalThreadId() const {
        return m_item.get_local_linear_id();
    }

    inline auto getLocalThreadIdX() const { return m_item.get_local_id(0); }

    inline auto getGlobalThreadId() const {
        return m_item.get_global_linear_id();
    }

    inline auto getGlobalThreadIdX() const { return m_item.get_global_id(0); }

    inline auto getBlockIdX() const { return m_item.get_group(0); }

    inline auto getBlockDimX() const { return m_item.get_local_range(0); }

    inline auto getGridDimX() const { return m_item.get_global_range(0); }

    private:
    /// Item object coming from the SYCL kernel
    const ::sycl::nd_item<dimensions>& m_item;

};  // struct thread_id

/// Verify that @c traccc::sycl::details::thread_id fulfills the
/// @c traccc::device::concepts::thread_id1 concept.
static_assert(traccc::device::concepts::thread_id1<thread_id<1>>);
static_assert(traccc::device::concepts::thread_id1<thread_id<2>>);
static_assert(traccc::device::concepts::thread_id1<thread_id<3>>);

}  // namespace traccc::sycl::details
