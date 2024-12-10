/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "pool_memory_resource_impl.hpp"

#include "../../utils/integer_math.hpp"
#include "vecmem/memory/details/is_aligned.hpp"
#include "vecmem/utils/debug.hpp"

// System include(s).
#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>

/// Helper macro for implementing the @c check_valid function
#define CHECK_VALID(EXP)                            \
    if (EXP) {                                      \
        std::ostringstream msg;                     \
        msg << __FILE__ << ":" << __LINE__          \
            << " Invalid pool option(s): " << #EXP; \
        throw std::invalid_argument(msg.str());     \
    }

namespace vecmem::details {
namespace {

/// Function checking whether a given set of options are valid/consistent
///
/// @param opts The options to check
///
void check_valid(const pool_memory_resource::options& opts) {

    CHECK_VALID(!vecmem::details::is_power_of_2(opts.smallest_block_size));
    CHECK_VALID(!vecmem::details::is_power_of_2(opts.largest_block_size));
    CHECK_VALID(!vecmem::details::is_power_of_2(opts.alignment));

    CHECK_VALID((opts.max_bytes_per_chunk == 0) ||
                (opts.max_blocks_per_chunk == 0));
    CHECK_VALID((opts.smallest_block_size == 0) ||
                (opts.largest_block_size == 0));

    CHECK_VALID(opts.min_blocks_per_chunk > opts.max_blocks_per_chunk);
    CHECK_VALID(opts.min_bytes_per_chunk > opts.max_bytes_per_chunk);

    CHECK_VALID(opts.smallest_block_size > opts.largest_block_size);

    CHECK_VALID((opts.min_blocks_per_chunk * opts.smallest_block_size) >
                opts.max_bytes_per_chunk);
    CHECK_VALID((opts.min_blocks_per_chunk * opts.largest_block_size) >
                opts.max_bytes_per_chunk);

    CHECK_VALID((opts.max_blocks_per_chunk * opts.largest_block_size) <
                opts.min_bytes_per_chunk);
    CHECK_VALID((opts.max_blocks_per_chunk * opts.smallest_block_size) <
                opts.min_bytes_per_chunk);

    CHECK_VALID(opts.alignment > opts.smallest_block_size);
}

}  // namespace

pool_memory_resource_impl::pool_memory_resource_impl(
    memory_resource& upstream, const pool_memory_resource::options& opts)
    : m_upstream(upstream),
      m_options(opts),
      m_smallest_block_log2(
          vecmem::details::log2_ri(opts.smallest_block_size)) {

    check_valid(opts);
    m_pools.resize(vecmem::details::log2_ri(m_options.largest_block_size) -
                   m_smallest_block_log2 + 1);
    VECMEM_DEBUG_MSG(5, "Created %lu pools", m_pools.size());
}

pool_memory_resource_impl::~pool_memory_resource_impl() {

    // Deallocate memory allocated for the buckets.
    for (chunk_descriptor& chunk : m_allocated) {
        m_upstream.get().deallocate(chunk.pointer, chunk.size,
                                    m_options.alignment);
    }

    // Deallocate the cached oversized/overaligned memory.
    for (oversized_block_descriptor& block : m_cached_oversized) {
        m_upstream.get().deallocate(block.pointer, block.size, block.alignment);
    }
}

void* pool_memory_resource_impl::allocate(std::size_t bytes,
                                          std::size_t alignment) {

    // Adjust the requested size to the minimum.
    bytes = std::max(bytes, m_options.smallest_block_size);
    assert(vecmem::details::is_power_of_2(alignment));

    // An oversized and/or overaligned allocation requested; needs to be
    // allocated separately.
    if ((bytes > m_options.largest_block_size) ||
        (alignment > m_options.alignment)) {

        // Create the descriptor of the block that we need.
        oversized_block_descriptor oversized;
        oversized.size = bytes;
        oversized.alignment = alignment;

        // If caching is allowed and the cache is not empty, try to find an
        // existing block that can be used.
        if (m_options.cache_oversized &&
            (m_cached_oversized.empty() == false)) {

            // Look for the first cached element that is big enough.
            auto it = std::lower_bound(m_cached_oversized.begin(),
                                       m_cached_oversized.end(), oversized);

            // If the size is bigger than the requested size by a factor
            // bigger than or equal to the specified cutoff for size,
            // allocate a new block.
            if (it != m_cached_oversized.end()) {
                const std::size_t size_factor = it->size / bytes;
                if (size_factor >= m_options.cached_size_cutoff_factor) {
                    it = m_cached_oversized.end();
                }
            }

            // Make sure that the alignment of the cached block is appropriate.
            if ((it != m_cached_oversized.end()) &&
                (it->alignment < alignment)) {
                it = std::find_if(
                    it + 1, m_cached_oversized.end(),
                    [alignment](const oversized_block_descriptor& desc) {
                        return desc.alignment >= alignment;
                    });
                ;
            }

            // If the alignment is bigger than the requested one by a factor
            // bigger than or equal to the specified cutoff for alignment,
            // allocate a new block.
            if (it != m_cached_oversized.end()) {
                const std::size_t alignment_factor = it->alignment / alignment;
                if (alignment_factor >=
                    m_options.cached_alignment_cutoff_factor) {
                    it = m_cached_oversized.end();
                }
            }

            // If a cached block was found, use it.
            if (it != m_cached_oversized.end()) {
                void* result = it->pointer;
                m_cached_oversized.erase(it);
                return result;
            }
        }

        // No fitting cached block found; allocate a new one that's just up to
        // the specs.
        oversized.pointer = m_upstream.get().allocate(bytes, alignment);
        m_oversized.push_back(oversized);
        return oversized.pointer;
    }

    // The request is NOT for oversized and/or overaligned memory
    // allocate a block from an appropriate bucket.
    const std::size_t bytes_log2 = vecmem::details::log2_ri(bytes);
    const std::size_t bucket_idx = bytes_log2 - m_smallest_block_log2;
    pool& bucket = m_pools[bucket_idx];

    // If the free list of the bucket has no elements, allocate a new chunk
    // and split it into blocks pushed to the free list.
    if (bucket.free_blocks.empty()) {

        const std::size_t bucket_size = static_cast<std::size_t>(1)
                                        << bytes_log2;

        std::size_t n = bucket.previous_allocated_count;
        if (n == 0) {
            n = m_options.min_blocks_per_chunk;
            if (n < (m_options.min_bytes_per_chunk >> bytes_log2)) {
                n = m_options.min_bytes_per_chunk >> bytes_log2;
            }
        } else {
            n = n * 3 / 2;
            if (n > (m_options.max_bytes_per_chunk >> bytes_log2)) {
                n = m_options.max_bytes_per_chunk >> bytes_log2;
            }
            if (n > m_options.max_blocks_per_chunk) {
                n = m_options.max_blocks_per_chunk;
            }
        }

        bytes = n << bytes_log2;

        assert(n >= m_options.min_blocks_per_chunk);
        assert(n <= m_options.max_blocks_per_chunk);
        assert(bytes >= m_options.min_bytes_per_chunk);
        assert(bytes <= m_options.max_bytes_per_chunk);

        chunk_descriptor allocated;
        allocated.size = bytes;
        allocated.pointer =
            m_upstream.get().allocate(bytes, m_options.alignment);
        m_allocated.push_back(allocated);
        bucket.previous_allocated_count = n;

        for (std::size_t i = 0; i < n; ++i) {
            bucket.free_blocks.push_back(static_cast<void*>(
                static_cast<char*>(allocated.pointer) + i * bucket_size));
        }
    }

    // Use a block from the back of the bucket's free list.
    assert(bucket.free_blocks.empty() == false);
    void* ret = bucket.free_blocks.back();
    bucket.free_blocks.pop_back();
    return ret;
}

void pool_memory_resource_impl::deallocate(void* ptr, std::size_t bytes,
                                           std::size_t alignment) {

    // Adjust the requested size to the minimum.
    bytes = std::max(bytes, m_options.smallest_block_size);
    assert(vecmem::details::is_power_of_2(alignment));
    assert(vecmem::details::is_aligned(ptr, alignment));

    // The deallocated block is oversized and/or overaligned.
    if ((bytes > m_options.largest_block_size) ||
        (alignment > m_options.alignment)) {

        // Find the oversized allocation.
        auto it = std::find_if(m_oversized.begin(), m_oversized.end(),
                               [ptr](const oversized_block_descriptor& desc) {
                                   return desc.pointer == ptr;
                               });
        assert(it != m_oversized.end());
        oversized_block_descriptor oversized = *it;

        // If oversized allocations are to be cached, put the block into the
        // cache.
        if (m_options.cache_oversized) {
            auto position =
                std::lower_bound(m_cached_oversized.begin(),
                                 m_cached_oversized.end(), oversized);
            m_cached_oversized.insert(position, oversized);
            return;
        } else {
            // Otherwise forget about the block, and deallocate the memory.
            m_oversized.erase(it);
            m_upstream.get().deallocate(ptr, oversized.size,
                                        oversized.alignment);
            return;
        }
    }

    // Push the block at the end of the appropriate bucket's free list.
    const std::size_t n_log2 = vecmem::details::log2_ri(bytes);
    const std::size_t bucket_idx = n_log2 - m_smallest_block_log2;
    pool& bucket = m_pools[bucket_idx];
    bucket.free_blocks.push_back(ptr);
}

bool pool_memory_resource_impl::oversized_block_descriptor::operator<(
    const oversized_block_descriptor& other) const {
    return ((size < other.size) ||
            ((size == other.size) && (alignment < other.alignment)));
}

}  // namespace vecmem::details
