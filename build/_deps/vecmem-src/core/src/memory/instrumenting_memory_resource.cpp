/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/instrumenting_memory_resource.hpp"

#include "details/instrumenting_memory_resource_impl.hpp"
#include "details/memory_resource_impl.hpp"

namespace vecmem {

instrumenting_memory_resource::memory_event::memory_event(type t, std::size_t s,
                                                          std::size_t a,
                                                          void* p,
                                                          std::size_t ns)
    : m_type(t), m_size(s), m_align(a), m_ptr(p), m_time(ns) {}

instrumenting_memory_resource::instrumenting_memory_resource(
    memory_resource& upstream)
    : m_impl{std::make_unique<details::instrumenting_memory_resource_impl>(
          upstream)} {}

const std::vector<instrumenting_memory_resource::memory_event>&
instrumenting_memory_resource::get_events(void) const {

    return m_impl->get_events();
}

void instrumenting_memory_resource::add_pre_allocate_hook(
    std::function<void(std::size_t, std::size_t)> f) {

    m_impl->add_pre_allocate_hook(f);
}

void instrumenting_memory_resource::add_post_allocate_hook(
    std::function<void(std::size_t, std::size_t, void*)> f) {

    m_impl->add_post_allocate_hook(f);
}

void instrumenting_memory_resource::add_pre_deallocate_hook(
    std::function<void(void*, std::size_t, std::size_t)> f) {

    m_impl->add_pre_deallocate_hook(f);
}

VECMEM_MEMORY_RESOURCE_PIMPL_IMPL(instrumenting_memory_resource)

}  // namespace vecmem
