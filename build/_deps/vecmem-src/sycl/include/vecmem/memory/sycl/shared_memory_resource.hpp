/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/sycl/details/memory_resource_base.hpp"
#include "vecmem/vecmem_sycl_export.hpp"

namespace vecmem::sycl {

/// Memory resource shared between the host and a specific SYCL device
class shared_memory_resource final : public details::memory_resource_base {

public:
    /// Constructor on top of a user-provided queue
    VECMEM_SYCL_EXPORT
    shared_memory_resource(const queue_wrapper& queue = {});
    /// Destructor
    VECMEM_SYCL_EXPORT
    ~shared_memory_resource();

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Function performing the memory allocation
    VECMEM_SYCL_EXPORT
    virtual void* do_allocate(std::size_t nbytes,
                              std::size_t alignment) override final;

    /// @}

};  // class shared_memory_resource

}  // namespace vecmem::sycl
