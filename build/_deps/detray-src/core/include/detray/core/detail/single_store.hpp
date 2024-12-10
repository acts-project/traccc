/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/core/detail/container_buffers.hpp"
#include "detray/core/detail/container_views.hpp"
#include "detray/core/detail/data_context.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/detail/qualifiers.hpp"

// Vecmem include(s)
#include <vecmem/memory/memory_resource.hpp>

// System include(s)
#include <type_traits>

namespace detray {

/// @brief Wraps a vector-like container and implements a data store around it.
///
/// @tparam T The type of the collection data, e.g. transforms
/// @tparam container_t The type of container to use for the data collection.
/// @tparam context_t the context with which to retrieve the correct data.
template <typename T, template <typename...> class container_t = dvector,
          typename context_t = empty_context>
class single_store {

    public:
    /// Underlying container type that can handle vecmem views
    using base_type = container_t<T>;
    using size_type = typename base_type::size_type;
    using value_type = typename base_type::value_type;
    using iterator = typename base_type::iterator;
    using const_iterator = typename base_type::const_iterator;
    using context_type = context_t;

    /// How to find data in the store
    /// @{
    using link_type = dindex;
    using single_link = dindex;
    using range_link = dindex_range;
    /// @}

    /// Vecmem view types
    using view_type = detail::get_view_t<container_t<T>>;
    using const_view_type = detail::get_view_t<const container_t<T>>;
    using buffer_type = detail::get_buffer_t<container_t<T>>;

    /// Empty container
    constexpr single_store() = default;

    // Delegate constructors to container, which handles the memory

    /// Copy construct from element types
    constexpr explicit single_store(const T &arg) : m_container(arg) {}

    /// Construct with a specific memory resource @param resource
    /// (host-side only)
    template <typename allocator_t = vecmem::memory_resource>
    requires(std::derived_from<allocator_t, std::pmr::memory_resource>)
        DETRAY_HOST explicit single_store(allocator_t &resource)
        : m_container(&resource) {}

    /// Copy Construct with a specific memory resource @param resource
    /// (host-side only)
    template <typename allocator_t = vecmem::memory_resource,
              typename C = container_t<T>>
    requires(std::is_same_v<C, std::vector<T>>
                 &&std::derived_from<allocator_t, std::pmr::memory_resource>)
        DETRAY_HOST single_store(allocator_t &resource, const T &arg)
        : m_container(&resource, arg) {}

    /// Construct from the container @param view . Mainly used device-side.
    template <concepts::device_view container_view_t>
    DETRAY_HOST_DEVICE explicit single_store(container_view_t &view)
        : m_container(view) {}

    /// @returns a pointer to the underlying container - const
    DETRAY_HOST_DEVICE
    constexpr auto data() const noexcept -> const base_type * {
        return &m_container;
    }

    /// @returns a pointer to the underlying container - non-const
    DETRAY_HOST_DEVICE
    constexpr auto data() noexcept -> base_type * { return &m_container; }

    /// @returns the size of the underlying container
    DETRAY_HOST_DEVICE
    constexpr auto size(const context_type & /*ctx*/ = {}) const noexcept
        -> dindex {
        return static_cast<dindex>(m_container.size());
    }

    /// @returns true if the underlying container is empty
    DETRAY_HOST_DEVICE
    constexpr auto empty(const context_type & /*ctx*/ = {}) const noexcept
        -> bool {
        return m_container.empty();
    }

    /// @returns the collections iterator at the start position
    DETRAY_HOST_DEVICE
    constexpr auto begin(const context_type & /*ctx*/ = {}) const {
        return m_container.begin();
    }

    /// @returns the collections iterator sentinel
    DETRAY_HOST_DEVICE
    constexpr auto end(const context_type & /*ctx*/ = {}) const {
        return m_container.end();
    }

    /// @returns access to the underlying container - const
    DETRAY_HOST_DEVICE
    constexpr auto get(const context_type & /*ctx*/) const noexcept
        -> const base_type & {
        return m_container;
    }

    /// @returns access to the underlying container - non-const
    DETRAY_HOST_DEVICE
    constexpr auto get(const context_type & /*ctx*/) noexcept -> base_type & {
        return m_container;
    }

    /// @returns context based access to an element (also range checked)
    DETRAY_HOST_DEVICE
    constexpr auto at(const dindex i,
                      const context_type &ctx = {}) const noexcept
        -> const T & {
        [[maybe_unused]] context_type tmp_ctx{
            ctx};  // Temporary measure to avoid warnings
        return m_container.at(i);
    }

    /// @returns context based access to an element (also range checked)
    DETRAY_HOST_DEVICE
    constexpr auto at(const dindex i, const context_type &ctx = {}) noexcept
        -> T & {
        [[maybe_unused]] context_type tmp_ctx{
            ctx};  // Temporary measure to avoid warnings
        return m_container.at(i);
    }

    /// Removes and destructs all elements in the container.
    DETRAY_HOST void clear(const context_type & /*ctx*/) {
        m_container.clear();
    }

    /// Reserve memory of size @param n for a given geometry context
    DETRAY_HOST void reserve(std::size_t n, const context_type & /*ctx*/) {
        m_container.reserve(n);
    }

    /// Resize the underlying container to @param n for a given geometry context
    DETRAY_HOST void resize(std::size_t n, const context_type & /*ctx*/) {
        m_container.resize(n);
    }

    /// Add a new element to the collection - copy
    ///
    /// @tparam U type that can be converted to T
    ///
    /// @param arg the constructor argument
    ///
    /// @note in general can throw an exception
    template <typename U>
    DETRAY_HOST constexpr auto push_back(
        const U &arg, const context_type & /*ctx*/ = {}) noexcept(false)
        -> void {
        m_container.push_back(arg);
    }

    /// Add a new element to the collection - move
    ///
    /// @tparam U type that can be converted to T
    ///
    /// @param arg the constructor argument
    ///
    /// @note in general can throw an exception
    template <typename U>
    DETRAY_HOST constexpr auto push_back(
        U &&arg, const context_type & /*ctx*/ = {}) noexcept(false) -> void {
        m_container.push_back(std::forward<U>(arg));
    }

    /// Add a new element to the collection in place
    ///
    /// @tparam Args are the types of the constructor arguments
    ///
    /// @param args is the list of constructor arguments
    ///
    /// @note in general can throw an exception
    template <typename... Args>
    DETRAY_HOST constexpr decltype(auto) emplace_back(
        const context_type & /*ctx*/ = {}, Args &&... args) noexcept(false) {
        return m_container.emplace_back(std::forward<Args>(args)...);
    }

    /// Insert another collection - copy
    ///
    /// @tparam U type that can be converted to T
    ///
    /// @param new_data is the new collection to be added
    ///
    /// @note in general can throw an exception
    template <typename U>
    DETRAY_HOST auto insert(container_t<U> &new_data,
                            const context_type & /*ctx*/ = {}) noexcept(false)
        -> void {
        m_container.reserve(m_container.size() + new_data.size());
        m_container.insert(m_container.end(), new_data.begin(), new_data.end());
    }

    /// Insert another collection - move
    ///
    /// @tparam U type that can be converted to T
    ///
    /// @param new_data is the new collection to be added
    ///
    /// @note in general can throw an exception
    template <typename U>
    DETRAY_HOST auto insert(container_t<U> &&new_data,
                            const context_type & /*ctx*/ = {}) noexcept(false)
        -> void {
        m_container.reserve(m_container.size() + new_data.size());
        m_container.insert(m_container.end(),
                           std::make_move_iterator(new_data.begin()),
                           std::make_move_iterator(new_data.end()));
    }

    /// Append another store to the current one
    ///
    /// @param other The other container
    ///
    /// @note in general can throw an exception
    DETRAY_HOST void append(single_store &other,
                            const context_type &ctx = {}) noexcept(false) {
        insert(other.m_container, ctx);
    }

    /// Append another store to the current one - move
    ///
    /// @param other The other container
    ///
    /// @note in general can throw an exception
    DETRAY_HOST void append(single_store &&other,
                            const context_type &ctx = {}) noexcept(false) {
        insert(std::move(other.m_container), ctx);
    }

    /// @return the view on the underlying container - non-const
    DETRAY_HOST auto get_data() -> view_type {
        return detray::get_data(m_container);
    }

    /// @return the view on the underlying container - const
    DETRAY_HOST auto get_data() const -> const_view_type {
        return detray::get_data(m_container);
    }

    private:
    /// The underlying container implementation
    base_type m_container;
};

}  // namespace detray
