/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/memory_resource.hpp>

namespace traccc {
class caching_memory_resource final : public vecmem::memory_resource {
    public:
    caching_memory_resource(vecmem::memory_resource& base_mr,
                            std::size_t threshold);

    void* do_allocate(std::size_t bytes, std::size_t alignment) final;
    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) final;
    bool do_is_equal(const vecmem::memory_resource& other) const noexcept final;

    private:
    vecmem::memory_resource& m_base_mr;
    std::unique_ptr<vecmem::binary_page_memory_resource> m_caching_mr;
    std::size_t m_threshold;
};

std::unique_ptr<vecmem::memory_resource> make_caching_memory_resource(
    vecmem::memory_resource& base_mr, std::size_t threshold);
}  // namespace traccc
