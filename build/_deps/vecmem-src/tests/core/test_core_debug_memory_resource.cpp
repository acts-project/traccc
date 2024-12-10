/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include "vecmem/memory/debug_memory_resource.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/memory_resource.hpp"

namespace {
class broken_double_allocate_memory_resource final
    : public vecmem::memory_resource {

public:
    broken_double_allocate_memory_resource(memory_resource& upstream)
        : m_upstream(upstream) {}

    ~broken_double_allocate_memory_resource() {
        if (m_ptr != nullptr) {
            m_upstream.deallocate(m_ptr, m_size);
        }
    }

private:
    virtual void* do_allocate(std::size_t s, std::size_t a) override {
        if (m_ptr == nullptr) {
            m_size = s;
            return (m_ptr = m_upstream.allocate(s, a));
        } else {
            return m_ptr;
        }
    }

    virtual void do_deallocate(void*, std::size_t, std::size_t) override {}

    virtual bool do_is_equal(
        const vecmem::memory_resource&) const noexcept override {
        return false;
    }

    vecmem::memory_resource& m_upstream;

    void* m_ptr = nullptr;
    std::size_t m_size;
};
}  // namespace

TEST(core_debug_memory_resource_test, double_allocate) {
    vecmem::host_memory_resource ups;
    broken_double_allocate_memory_resource bro(ups);
    vecmem::debug_memory_resource res(bro);

    void* p = nullptr;

    EXPECT_NO_THROW(p = res.allocate(1024));
    EXPECT_THROW(p = res.allocate(1024), std::logic_error);
    EXPECT_NO_THROW(res.deallocate(p, 1024));

    EXPECT_NO_THROW(p = res.allocate(1024));
    EXPECT_THROW(p = res.allocate(1024), std::logic_error);
    EXPECT_NO_THROW(res.deallocate(p, 1024));
}

TEST(core_debug_memory_resource_test, invalid_deallocate) {
    vecmem::host_memory_resource ups;
    vecmem::debug_memory_resource res(ups);

    void* p = nullptr;

    EXPECT_NO_THROW(p = res.allocate(1024));
    EXPECT_NO_THROW(res.deallocate(p, 1024));

    EXPECT_NO_THROW(p = res.allocate(1024));
    EXPECT_THROW(res.deallocate(p, 2048), std::logic_error);
    EXPECT_NO_THROW(res.deallocate(p, 1024));

    EXPECT_NO_THROW(p = res.allocate(1024, 32));
    EXPECT_THROW(res.deallocate(p, 1024, 64), std::logic_error);
    EXPECT_NO_THROW(res.deallocate(p, 1024, 32));
}

TEST(core_debug_memory_resource_test, double_deallocate) {
    vecmem::host_memory_resource ups;
    vecmem::debug_memory_resource res(ups);

    void* p = nullptr;

    EXPECT_NO_THROW(p = res.allocate(1024));
    EXPECT_NO_THROW(res.deallocate(p, 1024));
    EXPECT_THROW(res.deallocate(p, 1024), std::logic_error);
}
