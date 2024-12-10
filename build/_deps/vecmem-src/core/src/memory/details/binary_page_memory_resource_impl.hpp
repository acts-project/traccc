/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/memory/unique_ptr.hpp"

// System include(s).
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace vecmem::details {

/// Implementation of @c vecmem::binary_page_memory_resource
struct binary_page_memory_resource_impl {
    /**
     * @brief The minimum size (log_2) of superpages in our buddy allocator.
     *
     * The default value of 20 indicates the the default size is equal to
     * 2^20=1048576 bytes.
     *
     * @note Pages can (counterintuitively) be smaller than this value. This
     * happens if an allocation is so small that the allocation size plus the
     * delta is smaller than the minimum page size. The minimum superpage size
     * indicates the size of the superpage above which we will not
     * optimistically overallocate.
     */
    static constexpr std::size_t min_superpage_size = 20;

    /**
     * @brief The maximum difference (log_2) between the size of the superpage
     * and its smallest page.
     *
     * The default value of 8 indicates that there are at most 8 orders of
     * magnitude (log_2) between the size of the superpage, and the smallest
     * page in it.
     */
    static constexpr std::size_t delta_superpage_size = 8;

    /**
     * @brief The minimum size (log_2) of pages in our buddy allocator.
     *
     * The default value of 2 indicates that the size is equal to 2^8=256
     * bytes.
     */
    static constexpr std::size_t min_page_size = 8;

    /// Constructor, on top of another memory resource
    binary_page_memory_resource_impl(memory_resource &upstream);

    /**
     * @brief The different possible states a page can be in.
     *
     * We define three different page states. An OCCUPIED state is non-free,
     * and used directly (thus it is not split). A VACANT page is not split
     * and unused. A SPLIT page is split in two, and has two children pages.
     * Non-extant pages do not exist, because their parent is not split.
     */
    enum class page_state { OCCUPIED, VACANT, SPLIT, NON_EXTANT };

    /**
     * @brief Container for superpages in our buddy allocator.
     *
     * Super pages are large, contigous allocations, which are split into
     * smaller pieces.
     */
    struct superpage {
        /**
         * @brief Construct a superpage with a given size and upstream
         * resource.
         */
        superpage(std::size_t, memory_resource &);

        /**
         * @brief Return the total number of pages in the superpage.
         */
        std::size_t total_pages() const;

        /**
         * @brief Size (log_2) of the entire allocation represented by this
         * superpage.
         */
        std::size_t m_size;

        /**
         * @brief The size of the smallest page in this superpage.
         */
        std::size_t m_min_page_size;

        /**
         * @brief Total number of pages in this superpage.
         */
        std::size_t m_num_pages;

        /**
         * @brief Array of pages, remembering that this always resides in
         * host-accessible memory.
         */
        std::unique_ptr<page_state[]> m_pages;

        /**
         * @brief The actual allocation, which is just a byte pointer. This
         * is potentially host-inaccessible.
         */
        unique_alloc_ptr<std::byte[]> m_memory;
    };

    /**
     * @brief Helper class to refer to pages in superpages.
     *
     * We identify individual pages as a tuple of the superpage and their index
     * in that superpage. This class exists to more ergonomically work with
     * these tuples, providing a variety of helper methods.
     */
    struct page_ref {
    public:
        /**
         * @brief Delete the meaningless default constructor.
         */
        page_ref() = delete;

        /**
         * @brief Default implementation of copy constructor.
         */
        page_ref(const page_ref &) = default;

        /**
         * @brief Default implementation of move constructor.
         */
        page_ref(page_ref &&) = default;

        /**
         * @brief Construct a page from a superpage and an index into that
         * superpage.
         */
        page_ref(superpage &, std::size_t);

        /**
         * @brief Default implementation of move assignment.
         */
        page_ref &operator=(page_ref &&) = default;

        /**
         * @brief Equality operator of pages.
         */
        bool operator==(const page_ref &) const;

        /**
         * @brief Inquality operator of pages.
         */
        bool operator!=(const page_ref &) const;

        /**
         * @brief Return the size (log_2) of the page referenced.
         */
        std::size_t get_size() const;

        /**
         * @brief Return the state of the page referenced.
         */
        page_state get_state() const;

        /**
         * @brief Return the beginning of the address space represented by this
         * page.
         */
        void *get_addr() const;

        /**
         * @brief Obtain a reference to this page's left child.
         */
        page_ref left_child() const;

        /**
         * @brief Obtain a reference to this page's left child.
         */
        page_ref right_child() const;

        /**
         * @brief Obtain a reference to this page's parent, if such a node
         * exists.
         */
        std::optional<page_ref> parent() const;

        /**
         * @brief Obtain a reference to this page's sibling, if such a node
         * exists.
         */
        std::optional<page_ref> sibling() const;

        /**
         * @brief Unsplit the current page, potentially unsplitting its
         * children, too.
         */
        void unsplit();

        /**
         * @brief Split the current page.
         */
        void split();

        /**
         * @brief Change page state from vacant to occupied.
         */
        void change_state_vacant_to_occupied();

        /**
         * @brief Change page state from occupied to vacant.
         */
        void change_state_occupied_to_vacant();

        /**
         * @brief Change page state from non-extant to vacant.
         */
        void change_state_non_extant_to_vacant();

        /**
         * @brief Change page state from vacant to non-extant.
         */
        void change_state_vacant_to_non_extant();

        /**
         * @brief Change page state from vacant to split.
         */
        void change_state_vacant_to_split();

        /**
         * @brief Change page state from split to vacant.
         */
        void change_state_split_to_vacant();

        /**
         * @brief Get the page index in the superpage.
         */
        std::size_t get_index();

    private:
        std::reference_wrapper<superpage> m_superpage;
        std::size_t m_page;
    };

    /// @name Functions implementing the @c vecmem::memory_resource interface
    /// @{

    /// Allocate a blob of memory
    void *allocate(std::size_t size, std::size_t align);
    /// De-allocate a previously allocated memory blob
    void deallocate(void *p, std::size_t size, std::size_t align);

    /// @}

    /**
     * @brief Find the smallest free page that could fit the requested size.
     *
     * Note that this method might return split pages if both children are
     * free. In that case, the page should first be unsplit. In some cases,
     * the returned page might be (significantly) larger than the request,
     * and should be split before allocating.
     */
    std::optional<page_ref> find_free_page(std::size_t);

    /**
     * @brief Perform an upstream allocation.
     *
     * This method performs an allocation through the upstream memory
     * resource and immediately creates a page to represent this new chunk
     * of memory.
     */
    void allocate_upstream(std::size_t);

    memory_resource &m_upstream;
    std::vector<superpage> m_superpages;

};  // struct binary_page_memory_resource_impl

}  // namespace vecmem::details
