/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"

// System include(s).
#include <cstddef>
#include <limits>
#include <set>

namespace vecmem::details {

/// Implementation backend for @c vecmem::details::arena_page_memory_resource
class arena_memory_resource_impl {

public:
    // default initial size for the arena
    static constexpr std::size_t default_initial_size =
        std::numeric_limits<std::size_t>::max();
    // default maximum size for the arena
    static constexpr std::size_t default_maximum_size =
        std::numeric_limits<std::size_t>::max();
    // reserved memory that should not be allocated (64 MiB)
    static constexpr std::size_t reserverd_size = 1u << 26u;

    // Construct an `arena`
    //
    // @param[in] arena the arena from withc to allocate
    // superblocks
    explicit arena_memory_resource_impl(std::size_t initial_size,
                                        std::size_t maximum_size,
                                        memory_resource& mm);

    ~arena_memory_resource_impl();

    // Allocates memory of size at least `bytes`
    //
    // @param[in] bytes the size in bytes of the allocation
    // @param[in] alignment the alignment of the allocation (unused)
    // @return void* pointer to the newly allocated memory
    void* allocate(std::size_t bytes, std::size_t alignment = 0);

    // Deallocate memory pointed to by `p`, and keeping all free superblocks.
    // return the block to the set that have the free blocks.
    //
    // @param[in] p the pointer of the memory
    // @param[in] bytes the size in bytes of the deallocation
    // @param[in] alignment the alignment of the deallocation (unused)
    // @return if the allocation was found, false otherwise
    bool deallocate(void* p, std::size_t bytes, std::size_t alignment = 0);

private:
    /// Representation of a memory block
    class block {

    public:
        // construct a default block.
        block() = default;

        // construct a block given a pointer and size.
        //
        // @param[in] pointer the address for the beginning of the block.
        // @param[in] size the size of the block
        block(void* pointer, std::size_t size);

        // returns the underlying pointer
        void* pointer() const;

        // returns the size of the block
        std::size_t size() const;

        // returns true if this block is valid (non-null), false otherwise
        bool is_valid() const;

        // returns true if this block is a Superblock, false otherwise
        bool is_superblock() const;

        // verifies wheter this block can be merged to the beginning of block b
        //
        // @param[in] b the block to check for contiguity
        // @return true if this block's `pointer` + `size` == `b.ptr`, and `not
        // b.isHead`, false otherwise
        bool is_contiguous_before(block const& b) const;

        // is this block large enough to fit that size of bytes?
        //
        // @param[in] size_of_bytes the size in bytes to check for fit
        // @return true if this block is at least size_of_bytes
        bool fits(std::size_t size_of_bytes) const;

        // split this block into two by the given size
        //
        // @param[in] size the size in bytes of the first block
        // @return std::pair<block, block> a pair of blocks split by size
        std::pair<block, block> split(std::size_t size) const;

        // coalesce two contiguos blocks into one, this->is_contiguous_before(b)
        // must be true
        //
        // @param[in] b block to merge
        // @return block the merged block
        block merge(block const& b) const;

        // used by std::set to compare blocks
        bool operator<(block const& b) const;

    private:
        char* pointer_{};     // raw memory pointer
        std::size_t size_{};  // size in bytes
    };

    static block first_fit(std::set<block>& free_blocks, std::size_t size);
    static block coalesce_block(std::set<block>& free_blocks, block const& b);

    // @brief Get an available memory block of at least `size` bytes.
    //
    // @param[in] size The number of bytes to allocate.
    // @return block A block of memory of at least `size` bytes.
    block get_block(std::size_t size);

    // Allocate space from upstream to supply the arena and return a superblock.
    //
    // @return block A superblock.
    block expand_arena(std::size_t size);

    // Finds, frees and returns the block associated with pointer `p`.
    //
    // @param[in] p The pointer to the memory to free.
    // @param[in] size The size of the memory to free. Must be equal to the
    // original allocation size. return The (now freed) block associated with
    // `p`. The caller is expected to return the block to the arena.
    block free_block(void* p, std::size_t size) noexcept;

    memory_resource& mm_;
    // The size of superblocks to allocate in case of is necessarry
    std::size_t size_superblocks_{};
    // The maximum size of the arena
    std::size_t maximum_size_;
    // The current size of the arena
    std::size_t current_size_{};
    // Address-ordered set of free blocks
    std::set<block> free_blocks_;
    std::set<block> allocated_blocks_;

};  // class arena_memory_resource_impl

}  // namespace vecmem::details
