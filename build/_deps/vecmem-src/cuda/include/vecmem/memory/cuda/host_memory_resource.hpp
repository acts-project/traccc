/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_cuda_export.hpp"

namespace vecmem::cuda {

/**
 * @brief Memory resource that wraps page-locked CUDA host allocation.
 *
 * This is an allocator-type memory resource that allocates CUDA host
 * memory, which is page-locked by default to allow faster transfer to the
 * CUDA devices.
 */
class host_memory_resource final : public memory_resource {

public:
    /// Default constructor
    VECMEM_CUDA_EXPORT
    host_memory_resource();
    /// Destructor
    VECMEM_CUDA_EXPORT
    ~host_memory_resource();

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Allocate page-locked host memory
    VECMEM_CUDA_EXPORT
    virtual void* do_allocate(std::size_t, std::size_t) override final;
    /// De-allocate a previously allocated page-locked memory block
    VECMEM_CUDA_EXPORT
    virtual void do_deallocate(void* p, std::size_t,
                               std::size_t) override final;
    /// Compares @c *this for equality with @c other
    VECMEM_CUDA_EXPORT
    virtual bool do_is_equal(
        const memory_resource& other) const noexcept override final;

    /// @}

};  // class host_memory_resource

}  // namespace vecmem::cuda
