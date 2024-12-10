/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <atomic>

#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_core_export.hpp"

#if defined(VECMEM_HAVE_PMR_MEMORY_RESOURCE)
namespace std::pmr {
#else
namespace std::experimental {
inline namespace fundamentals_v1 {
namespace pmr {
#endif

namespace {

std::atomic<memory_resource*>& default_resource() noexcept {
    static std::atomic<memory_resource*> res{new_delete_resource()};
    return res;
}

}  // namespace

VECMEM_CORE_EXPORT memory_resource* new_delete_resource() noexcept {
    // vecmem::host_memory_resource is not singleton, and does not check
    // equality via pointer equality, but is close enough for our purposes.
    static vecmem::host_memory_resource res{};
    return &res;
}

VECMEM_CORE_EXPORT memory_resource* get_default_resource() noexcept {
    return default_resource();
}

VECMEM_CORE_EXPORT memory_resource* set_default_resource(
    memory_resource* res) noexcept {

    memory_resource* new_res = res == nullptr ? new_delete_resource() : res;

    return default_resource().exchange(new_res);
}
#if defined(VECMEM_HAVE_PMR_MEMORY_RESOURCE)
}
#else
}  // namespace pmr
}  // namespace fundamentals_v1
}  // namespace std::experimental
#endif
