/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "arena_memory_resource_impl.hpp"

// System include(s).
#include <algorithm>
#include <cassert>

namespace {

[[maybe_unused]] inline constexpr bool is_supported_alignment(
    std::size_t alignment) {
    return (0 == (alignment & (alignment - 1)));
}

inline constexpr std::size_t align_up(std::size_t v,
                                      std::size_t align_bytes = 256) noexcept {
    // if the alignment is not support, the program will end
    assert(is_supported_alignment(align_bytes));
    return (v + (align_bytes - 1)) & ~(align_bytes - 1);
}

static constexpr std::size_t minimum_superblock_size = 1u << 18u;

}  // namespace

namespace vecmem::details {

arena_memory_resource_impl::block::block(void* pointer, std::size_t size)
    : pointer_(static_cast<char*>(pointer)), size_(size) {}

void* arena_memory_resource_impl::block::pointer() const {
    return this->pointer_;
}

std::size_t arena_memory_resource_impl::block::size() const {
    return this->size_;
}

bool arena_memory_resource_impl::block::is_valid() const {
    return this->pointer_ != nullptr;
}

bool arena_memory_resource_impl::block::is_contiguous_before(
    block const& b) const {
    return this->pointer_ + this->size_ == b.pointer_;
}

bool arena_memory_resource_impl::block::fits(std::size_t size_of_bytes) const {
    return this->size_ >= size_of_bytes;
}

std::pair<arena_memory_resource_impl::block, arena_memory_resource_impl::block>
arena_memory_resource_impl::block::split(std::size_t size) const {

    // assert condition of size_ >= size
    if (this->size_ > size) {
        return {{this->pointer_, size},
                {this->pointer_ + size, this->size_ - size}};
    } else {
        return {*this, {}};
    }
}

arena_memory_resource_impl::block arena_memory_resource_impl::block::merge(
    block const& b) const {

    // assert condition is_contiguous_before(b)
    return {this->pointer(), this->size_ + b.size_};
}

bool arena_memory_resource_impl::block::operator<(block const& b) const {
    return this->pointer_ < b.pointer_;
}

arena_memory_resource_impl::arena_memory_resource_impl(std::size_t initial_size,
                                                       std::size_t maximum_size,
                                                       memory_resource& mm)
    : mm_(mm), size_superblocks_{initial_size}, maximum_size_{maximum_size} {
    // assert unexpected null upstream pointer
    // assert initial arena size required to be a multiple of 256 bytes
    // assert maximum arena size required to be a multiple of 256 bytes

    if (initial_size == default_initial_size ||
        maximum_size == default_maximum_size) {
        if (initial_size == default_initial_size) {
            initial_size = align_up(initial_size / 2);
        }
        if (maximum_size == default_maximum_size) {
            this->maximum_size_ = default_maximum_size - reserverd_size;
        }
    }
    // initial size exceeds the maxium pool size
    this->free_blocks_.emplace(this->expand_arena(initial_size));
}

arena_memory_resource_impl::~arena_memory_resource_impl() {

    for (auto itr = allocated_blocks_.begin();
         itr != allocated_blocks_.end();) {
        auto aux = itr++;
        deallocate(aux->pointer(), aux->size());
    }

    for (auto itr = free_blocks_.begin(); itr != free_blocks_.end();) {
        auto aux = itr++;
        void* p = aux->pointer();
        std::size_t size = aux->size();
        mm_.deallocate(p, size);
    }
}

void* arena_memory_resource_impl::allocate(std::size_t bytes, std::size_t) {

    bytes = align_up(bytes);

    auto const b = get_block(bytes);
    this->allocated_blocks_.emplace(b);

    return b.pointer();
}

bool arena_memory_resource_impl::deallocate(void* p, std::size_t bytes,
                                            std::size_t) {

    bytes = align_up(bytes);

    auto const b = free_block(p, bytes);
    if (b.is_valid()) {
        coalesce_block(free_blocks_, b);
    }

    return b.is_valid();
}

arena_memory_resource_impl::block arena_memory_resource_impl::first_fit(
    std::set<block>& free_blocks, std::size_t size) {

    auto const iter =
        std::find_if(free_blocks.cbegin(), free_blocks.cend(),
                     [size](auto const& b) { return b.fits(size); });

    if (iter == free_blocks.cend()) {
        return {};
    } else {
        // remove the block from the freeList
        auto const b = *iter;
        auto const i = free_blocks.erase(iter);

        if (b.size() > size) {
            // split the block and put the remainder back.
            auto const split = b.split(size);
            free_blocks.insert(i, split.second);
            return split.first;
        } else {
            // b.size == size then return b
            return b;
        }
    }
}

arena_memory_resource_impl::block arena_memory_resource_impl::coalesce_block(
    std::set<block>& free_blocks, block const& b) {

    // return the given block in case is not valid
    if (!b.is_valid())
        return b;

    // find the right place (in ascending address order) to insert the block
    auto const next = free_blocks.lower_bound(b);
    auto const previous = next == free_blocks.cbegin() ? next : std::prev(next);

    // coalesce with neighboring blocks
    bool const merge_prev = previous->is_contiguous_before(b);
    bool const merge_next =
        next != free_blocks.cend() && b.is_contiguous_before(*next);

    block merged{};
    if (merge_prev && merge_next) {
        // if can merge with prev and next neighbors
        merged = previous->merge(b).merge(*next);

        free_blocks.erase(previous);

        auto const i = free_blocks.erase(next);
        free_blocks.insert(i, merged);
    } else if (merge_prev) {
        // if only can merge with prev neighbor
        merged = previous->merge(b);

        auto const i = free_blocks.erase(previous);
        free_blocks.insert(i, merged);
    } else if (merge_next) {
        // if only can merge with next neighbor
        merged = b.merge(*next);

        auto const i = free_blocks.erase(next);
        free_blocks.insert(i, merged);
    } else {
        // if can't be merge with either
        free_blocks.emplace(b);
        merged = b;
    }

    return merged;
}

arena_memory_resource_impl::block arena_memory_resource_impl::get_block(
    std::size_t size) {

    if (size < minimum_superblock_size) {
        auto const b = first_fit(this->free_blocks_, size);
        if (b.is_valid()) {
            return b;
        }
    }

    auto const superblock = expand_arena(size);
    coalesce_block(this->free_blocks_, superblock);
    return first_fit(this->free_blocks_, size);
}

arena_memory_resource_impl::block arena_memory_resource_impl::expand_arena(
    std::size_t size) {

    if (size > this->size_superblocks_)
        size = minimum_superblock_size;
    else {
        size = size_superblocks_;
    }
    std::pair<std::set<block>::iterator, bool> ret =
        free_blocks_.insert({mm_.allocate(size), size});

    current_size_ += size;
    return *(ret.first);
}

arena_memory_resource_impl::block arena_memory_resource_impl::free_block(
    void* p, std::size_t /*size*/) noexcept {

    auto const i =
        std::find_if(allocated_blocks_.cbegin(), allocated_blocks_.cend(),
                     [p](auto const& b) { return b.pointer() == p; });

    if (i == this->allocated_blocks_.end()) {
        return {};
    }

    auto const found = *i;

    this->allocated_blocks_.erase(i);

    return found;
}

}  // namespace vecmem::details
