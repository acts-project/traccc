/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/vecmem_sycl_export.hpp"

// System include(s).
#include <memory>

namespace vecmem::sycl {

// Forward declaration(s).
namespace details {
class opaque_queue;
}

/// Wrapper class for @c ::sycl::queue
///
/// It is necessary for passing around SYCL queue objects in code that should
/// not be directly exposed to the SYCL headers.
///
class queue_wrapper {

public:
    /// Construct a queue for the default device
    VECMEM_SYCL_EXPORT
    queue_wrapper();
    /// Wrap an existing @c ::sycl::queue object
    ///
    /// Without taking ownership of it!
    ///
    VECMEM_SYCL_EXPORT
    queue_wrapper(void* queue);

    /// Copy constructor
    VECMEM_SYCL_EXPORT
    queue_wrapper(const queue_wrapper& parent);
    /// Move constructor
    VECMEM_SYCL_EXPORT
    queue_wrapper(queue_wrapper&& parent);

    /// Destructor
    ///
    /// The destructor is declared and implemented explicitly as an empty
    /// function to make sure that client code would not try to generate it
    /// itself. Leading to problems about the symbols of
    /// @c vecmem::sycl::details::opaque_queue not being available in
    /// client code.
    ///
    VECMEM_SYCL_EXPORT
    ~queue_wrapper();

    /// Copy assignment
    VECMEM_SYCL_EXPORT
    queue_wrapper& operator=(const queue_wrapper& rhs);
    /// Move assignment
    VECMEM_SYCL_EXPORT
    queue_wrapper& operator=(queue_wrapper&& rhs);

    /// Access a typeless pointer to the managed @c ::sycl::queue object
    VECMEM_SYCL_EXPORT
    void* queue();
    /// Access a typeless pointer to the managed @c ::sycl::queue object
    VECMEM_SYCL_EXPORT
    const void* queue() const;

private:
    /// Bare pointer to the wrapped @c ::sycl::queue object
    void* m_queue;

    /// Smart pointer to the managed @c ::sycl::queue object
    std::unique_ptr<details::opaque_queue> m_managedQueue;

};  // class queue_wrapper

}  // namespace vecmem::sycl
