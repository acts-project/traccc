/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/host_memory_resource.hpp"

#include "vecmem/utils/debug.hpp"

// System include(s).
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <memory>

namespace vecmem {

host_memory_resource::host_memory_resource() = default;

host_memory_resource::~host_memory_resource() = default;

void *host_memory_resource::do_allocate(std::size_t bytes,
                                        std::size_t alignment) {

    if (bytes == 0) {
        return nullptr;
    }

#ifdef VECMEM_HAVE_STD_ALIGNED_ALLOC
    // Rely on std::aligned_alloc to give us properly aligned memory.
    // Note that std::aligned_alloc has a number of quirks about the parameters
    // that it would accept. Hence the tweaking of the alignment and size
    // values.
    const std::size_t align = std::max(alignment, alignof(std::max_align_t));
    const std::size_t size = bytes + (align - (bytes % align));
    void *ptr = std::aligned_alloc(align, size);
    // If the alignment failed for some reason, throw an exception.
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
#else
    // Ask for enough memory that will allow us to be able to align
    // the returned pointer for sure, **and** be able to store the
    // pointer to the beginning of the unaligned memory blob before
    // the aligned pointer.
    std::size_t alloc_bytes = bytes + alignment + sizeof(void *) - 1;
    void *unaligned_ptr = std::malloc(alloc_bytes);
    // If the alignment failed for some reason, throw an exception.
    if (unaligned_ptr == nullptr) {
        throw std::bad_alloc();
    }
    // Make a pointer that has space "before it" for the
    // pointer to the beginning of the unaligned space.
    void *ptr = static_cast<char *>(unaligned_ptr) + sizeof(void *);
    // The bytes available for alignment are a little less than what we
    // allocated.
    std::size_t align_bytes = alloc_bytes - sizeof(void *);
    // Perform the alignment.
    void *aligned_ptr = std::align(alignment, bytes, ptr, align_bytes);
    // The alignment was meant to succeed. If it didn't, we messed
    // something up in the code. So let's only check for that in Debug
    // builds.
    (void)aligned_ptr;
    assert(aligned_ptr != nullptr);
    // Store the unaligned pointer's value just before the aligned
    // pointer for later use.
    *(reinterpret_cast<void **>(static_cast<char *>(ptr) - sizeof(void *))) =
        unaligned_ptr;
#endif  // VECMEM_HAVE_STD_ALIGNED_ALLOC
    VECMEM_DEBUG_MSG(3,
                     "Allocated %lu bytes of (%lu aligned) host memory at %p",
                     bytes, alignment, ptr);
    return ptr;
}

void host_memory_resource::do_deallocate(void *ptr, std::size_t, std::size_t) {

    if (ptr == nullptr) {
        return;
    }

    VECMEM_DEBUG_MSG(3, "De-allocating host memory at %p", ptr);

#ifdef VECMEM_HAVE_STD_ALIGNED_ALLOC
    // We can directly de-allocate the memory that was given to us by
    // std::aligned_alloc earlier.
    std::free(ptr);
#else
    // Construct the (unaligned) pointer that we need to free.
    void *unaligned_ptr =
        *(reinterpret_cast<void **>(static_cast<char *>(ptr) - sizeof(void *)));
    std::free(unaligned_ptr);
#endif  // VECMEM_HAVE_STD_ALIGNED_ALLOC
}

bool host_memory_resource::do_is_equal(
    const memory_resource &other) const noexcept {
    /*
     * All malloc resources are equal to each other, because they have no
     * internal state. Of course they have a shared underlying state in the
     * form of the underlying C library memory manager, but that is not
     * relevant for us.
     */
    return (dynamic_cast<const host_memory_resource *>(&other) != nullptr);
}

}  // namespace vecmem
