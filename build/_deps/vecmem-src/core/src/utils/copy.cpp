/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include "vecmem/utils/copy.hpp"

#include "vecmem/utils/debug.hpp"

// System include(s).
#include <cstring>

namespace {
/// Empty/no-op implementation for @c vecmem::abstract_event
struct noop_event : public vecmem::abstract_event {
    virtual void wait() override {}
    virtual void ignore() override {}
};  // struct noop_event
}  // namespace

namespace vecmem {

void copy::do_copy(std::size_t size, const void* from_ptr, void* to_ptr,
                   type::copy_type) const {

    // Perform a simple POSIX memory copy.
    ::memcpy(to_ptr, from_ptr, size);

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(1,
                     "Performed POSIX memory copy of %lu bytes from %p "
                     "to %p",
                     size, from_ptr, to_ptr);
}

void copy::do_memset(std::size_t size, void* ptr, int value) const {

    // Perform the POSIX memory setting operation.
    ::memset(ptr, value, size);

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(2, "Set %lu bytes to %i at %p with POSIX memset", size,
                     value, ptr);
}

copy::event_type copy::create_event() const {

    // Make a no-op event.
    return std::make_unique<noop_event>();
}

}  // namespace vecmem
