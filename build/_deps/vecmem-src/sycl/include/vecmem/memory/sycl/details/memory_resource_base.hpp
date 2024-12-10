/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/sycl/queue_wrapper.hpp"
#include "vecmem/vecmem_sycl_export.hpp"

/// @brief Namespace for types that should not be used directly by clients
namespace vecmem::sycl::details {

/// SYCL memory resource base class
///
/// This class is used as base by all of the oneAPI/SYCL memory resource
/// classes. It holds functionality that those classes all need.
///
class memory_resource_base : public memory_resource {

public:
    /// Constructor on top of a user-provided queue
    VECMEM_SYCL_EXPORT
    memory_resource_base(const queue_wrapper& queue = {});
    /// Destructor
    VECMEM_SYCL_EXPORT
    ~memory_resource_base();

protected:
    /// The queue that the allocations are made for/on
    queue_wrapper m_queue;

private:
    /// @name Function(s) implemented from @c vecmem::memory_resource
    /// @{

    /// Function performing the memory de-allocation
    VECMEM_SYCL_EXPORT
    virtual void do_deallocate(void* ptr, std::size_t nbytes,
                               std::size_t alignment) override final;

    /// Function comparing two memory resource instances
    VECMEM_SYCL_EXPORT
    virtual bool do_is_equal(
        const memory_resource& other) const noexcept override final;

    /// @}

};  // memory_resource_base

}  // namespace vecmem::sycl::details
