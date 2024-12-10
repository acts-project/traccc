/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_hip_export.hpp"

namespace vecmem::hip {

/**
 * @brief Memory resource that wraps managed HIP allocation.
 *
 * This is an allocator-type memory resource that allocates managed HIP
 * memory, which is accessible directly to devices as well as to the host.
 */
class managed_memory_resource final : public memory_resource {

public:
    /// Default constructor
    VECMEM_HIP_EXPORT
    managed_memory_resource();
    /// Destructor
    VECMEM_HIP_EXPORT
    ~managed_memory_resource();

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Allocate HIP managed memory
    VECMEM_HIP_EXPORT
    virtual void* do_allocate(std::size_t, std::size_t) override final;
    /// De-allocate a previously allocated managed memory block
    VECMEM_HIP_EXPORT
    virtual void do_deallocate(void* p, std::size_t,
                               std::size_t) override final;
    /// Compares @c *this for equality with @c other
    VECMEM_HIP_EXPORT
    virtual bool do_is_equal(
        const memory_resource& other) const noexcept override final;

    /// @}

};  // class managed_memory_resource

}  // namespace vecmem::hip
