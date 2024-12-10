/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/terminal_memory_resource.hpp"

// System include(s).
#include <stdexcept>

namespace vecmem {

terminal_memory_resource::terminal_memory_resource(void) {}

terminal_memory_resource::terminal_memory_resource(memory_resource &) {}

terminal_memory_resource::~terminal_memory_resource() = default;

void *terminal_memory_resource::do_allocate(std::size_t, std::size_t) {

    /*
     * Allocation always fails.
     */
    throw std::bad_alloc();
}

void terminal_memory_resource::do_deallocate(void *, std::size_t, std::size_t) {

    /*
     * Deallocation is a no-op.
     */
    return;
}

bool terminal_memory_resource::do_is_equal(
    const memory_resource &other) const noexcept {

    /*
     * All terminal resources are equal.
     */
    return dynamic_cast<const terminal_memory_resource *>(&other) != nullptr;
}

}  // namespace vecmem
