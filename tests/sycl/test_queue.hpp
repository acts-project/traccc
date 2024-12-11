/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// System include(s).
#include <memory>

namespace traccc::sycl {

/// Queue to use in the SYCL tests
class test_queue {

    public:
    /// Default constructor
    test_queue();
    /// Destructor
    ~test_queue();

    /// Get the SYCL queue
    queue_wrapper queue();

    private:
    /// Internal data type
    struct impl;

    /// Pointer to the internal data
    std::unique_ptr<impl> m_impl;

};  // struct test_queue

}  // namespace traccc::sycl
