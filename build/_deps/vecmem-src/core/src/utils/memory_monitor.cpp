/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/utils/memory_monitor.hpp"

// System include(s).
#include <algorithm>
#include <cassert>
#include <cmath>

namespace vecmem {

memory_monitor::memory_monitor(instrumenting_memory_resource& resource) {

    resource.add_post_allocate_hook(
        [this](std::size_t size, std::size_t align, void* ptr) {
            this->post_allocate(size, align, ptr);
        });
    resource.add_pre_deallocate_hook(
        [this](void* ptr, std::size_t size, std::size_t align) {
            this->pre_deallocate(ptr, size, align);
        });
}

std::size_t memory_monitor::total_allocation() const {

    return m_total_alloc;
}

std::size_t memory_monitor::outstanding_allocation() const {

    return m_outstanding_alloc;
}

std::size_t memory_monitor::average_allocation() const {

    return static_cast<std::size_t>(std::round(
        static_cast<double>(m_total_alloc) / static_cast<double>(m_n_alloc)));
}

std::size_t memory_monitor::maximal_allocation() const {

    return m_maximum_alloc;
}

void memory_monitor::post_allocate(std::size_t size, std::size_t, void* ptr) {

    // Don't do anything on failed allocations.
    if (ptr == nullptr) {
        return;
    }

    ++m_n_alloc;
    m_total_alloc += size;
    m_outstanding_alloc += size;
    m_maximum_alloc = std::max(m_outstanding_alloc, m_maximum_alloc);
}

void memory_monitor::pre_deallocate(void*, std::size_t size, std::size_t) {

    assert(m_outstanding_alloc >= size);
    m_outstanding_alloc -= size;
}

}  // namespace vecmem
