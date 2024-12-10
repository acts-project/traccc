/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/sycl/queue_wrapper.hpp"
#include "vecmem/vecmem_sycl_export.hpp"

// System include(s).
#include <memory>

namespace vecmem::sycl {
namespace details {
/// Object holding on to internal data for @c vecmem::sycl::async_copy
struct async_copy_data;
}  // namespace details

/// Specialisation of @c vecmem::copy for SYCL
///
/// Unlike @c vecmem::cuda::copy and @c vecmem::hip::copy, this object does
/// have a state. As USM memory operations in SYCL happen through a
/// @c ::sycl::queue object. So this object needs to point to a valid
/// queue object itself.
///
/// Different to @c vecmem::sycl::copy, this type performs operations
/// asynchronously, requiring users to introduce synchronisation points
/// explicitly into their code as needed.
///
class async_copy : public vecmem::copy {

public:
    /// Constructor on top of a user-provided queue
    VECMEM_SYCL_EXPORT
    async_copy(const queue_wrapper& queue);
    /// Destructor
    VECMEM_SYCL_EXPORT
    ~async_copy();

protected:
    /// Perform a memory copy using SYCL
    VECMEM_SYCL_EXPORT
    virtual void do_copy(std::size_t size, const void* from, void* to,
                         type::copy_type cptype) const override final;
    /// Fill a memory area using SYCL
    VECMEM_SYCL_EXPORT
    virtual void do_memset(std::size_t size, void* ptr,
                           int value) const override final;
    /// Create an event for synchronization
    VECMEM_SYCL_EXPORT
    virtual event_type create_event() const override final;

private:
    /// Internal data for the object
    std::unique_ptr<details::async_copy_data> m_data;

};  // class async_copy

}  // namespace vecmem::sycl
