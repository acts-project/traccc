/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/examples/utils/caching_memory_resource.hpp"

#include <vecmem/memory/binary_page_memory_resource.hpp>

namespace traccc {
caching_memory_resource::caching_memory_resource(
    vecmem::memory_resource& base_mr, std::size_t threshold)
    : m_base_mr(base_mr),
      m_caching_mr(
          std::make_unique<vecmem::binary_page_memory_resource>(m_base_mr)),
      m_threshold(threshold) {}

void* caching_memory_resource::do_allocate(std::size_t bytes,
                                           std::size_t alignment) {
    if (bytes >= m_threshold) {
        return m_base_mr.allocate(bytes, alignment);
    } else {
        return m_caching_mr->allocate(bytes, alignment);
    }
}

void caching_memory_resource::do_deallocate(void* p, std::size_t bytes,
                                            std::size_t alignment) {
    if (bytes >= m_threshold) {
        return m_base_mr.deallocate(p, bytes, alignment);
    } else {
        return m_caching_mr->deallocate(p, bytes, alignment);
    }
}

bool caching_memory_resource::do_is_equal(
    const vecmem::memory_resource& other) const noexcept {
    return (this == &other);
}

std::unique_ptr<vecmem::memory_resource> make_caching_memory_resource(
    vecmem::memory_resource& base_mr, std::size_t threshold) {
    return std::make_unique<caching_memory_resource>(base_mr, threshold);
}
}  // namespace traccc
