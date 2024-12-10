/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/containers.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/utils/bit_encoder.hpp"
#include "detray/utils/invalid_values.hpp"

// System include(s)
#include <array>
#include <cstdint>
#include <ostream>

namespace detray {

template <typename I>
DETRAY_HOST inline std::ostream& operator<<(std::ostream& os,
                                            const darray<I, 2>& r) {

    bool writeSeparator = false;
    for (auto i = 0u; i < r.size(); ++i) {
        if (writeSeparator) {
            os << ", ";
        }
        os << "[" << i << "]: " << r[i];
        writeSeparator = true;
    }
    return os;
}

namespace detail {

/// @brief Simple multi-index structure
///
/// @tparam DIM number of indices that are held by this type
/// @tparam index_t type of indices
template <typename index_t, std::size_t DIM>
struct multi_index {
    using index_type = index_t;

    std::array<index_t, DIM> indices{};

    /// @returns the number of conatined indices
    DETRAY_HOST_DEVICE
    constexpr static auto size() -> std::size_t { return DIM; }

    /// Elementwise access.
    DETRAY_HOST_DEVICE
    auto operator[](const std::size_t i) -> index_t& { return indices[i]; }
    DETRAY_HOST_DEVICE
    auto operator[](const std::size_t i) const -> const index_t& {
        return indices[i];
    }

    /// Equality operator @returns true if all bin indices match.
    bool operator==(const multi_index& rhs) const = default;

    DETRAY_HOST
    friend std::ostream& operator<<(std::ostream& os, const multi_index& mi) {

        bool writeSeparator = false;
        for (auto i = 0u; i < DIM; ++i) {
            if (writeSeparator) {
                os << ", ";
            }
            os << "[" << i << "]: " << mi[i];
            writeSeparator = true;
        }
        return os;
    }
};

/// @brief Ties an object type and an index into a container together.
///
/// @tparam id_type Represents the type of object that is being indexed
/// @tparam index_type The type of indexing needed for the indexed type's
///         container (e.g. single index, range, multiindex)
template <typename id_t, typename index_t,
          typename value_t = std::uint_least32_t, value_t id_mask = 0xf0000000,
          value_t index_mask = ~id_mask>
struct typed_index {

    using id_type = id_t;
    using index_type = index_t;
    using encoder = detail::bit_encoder<value_t>;

    constexpr typed_index() = default;

    DETRAY_HOST_DEVICE
    typed_index(const id_t id, const index_t idx) {
        encoder::template set_bits<id_mask>(m_value, static_cast<value_t>(id));
        encoder::template set_bits<index_mask>(m_value,
                                               static_cast<value_t>(idx));
    };

    /// @return the type id - const
    DETRAY_HOST_DEVICE
    constexpr auto id() const -> id_type {
        return static_cast<id_type>(
            encoder::template get_bits<id_mask>(m_value));
    }

    /// @return the index - const
    DETRAY_HOST_DEVICE
    constexpr auto index() const -> index_type {
        return static_cast<index_type>(
            encoder::template get_bits<index_mask>(m_value));
    }

    /// Set the link id.
    DETRAY_HOST_DEVICE
    constexpr typed_index& set_id(id_type id) {
        encoder::template set_bits<id_mask>(m_value, static_cast<value_t>(id));
        return *this;
    }

    /// Set the link index.
    DETRAY_HOST_DEVICE
    constexpr typed_index& set_index(index_type index) {
        encoder::template set_bits<index_mask>(m_value,
                                               static_cast<value_t>(index));
        return *this;
    }

    /// Comparison operators
    /// @{
    bool operator==(const typed_index&) const = default;

    /// Comparison operators
    DETRAY_HOST_DEVICE
    friend constexpr auto operator<=>(const typed_index lhs,
                                      const typed_index rhs) noexcept {
        const auto l{lhs.index()};
        const auto r{rhs.index()};
        if (l < r || (l == r && l < r)) {
            return std::strong_ordering::less;
        }
        if (l > r || (l == r && l > r)) {
            return std::strong_ordering::greater;
        }
        return std::strong_ordering::equivalent;
    }
    /// @}

    /// Arithmetic operators
    /// @{
    DETRAY_HOST_DEVICE
    friend typed_index operator+(const typed_index lhs, const typed_index rhs) {
        return typed_index{}.set_id(lhs.id()).set_index(lhs.index() +
                                                        rhs.index());
    }

    DETRAY_HOST_DEVICE
    friend typed_index operator+(const typed_index lhs,
                                 const index_type index) {
        return typed_index{}.set_id(lhs.id()).set_index(lhs.index() + index);
    }

    DETRAY_HOST_DEVICE
    friend typed_index operator-(const typed_index lhs, const typed_index rhs) {
        return typed_index{}.set_id(lhs.id()).set_index(lhs.index() -
                                                        rhs.index());
    }

    DETRAY_HOST_DEVICE
    friend typed_index operator-(const typed_index lhs,
                                 const index_type& index) {
        return typed_index{}.set_id(lhs.id()).set_index(lhs.index() - index);
    }

    DETRAY_HOST_DEVICE
    typed_index& operator+=(const typed_index rhs) {
        set_index(this->index() + rhs.index());
        return *this;
    }

    DETRAY_HOST_DEVICE
    typed_index& operator+=(const index_type index) {
        set_index(this->index() + index);
        return *this;
    }

    DETRAY_HOST_DEVICE
    typed_index& operator-=(const typed_index rhs) {
        set_index(this->index() - rhs.index());
        return *this;
    }

    DETRAY_HOST_DEVICE
    typed_index& operator-=(const index_type index) {
        set_index(this->index() + index);
        return *this;
    }
    /// @}

    /// Only make the prefix operator available
    DETRAY_HOST_DEVICE
    typed_index& operator++() {
        set_index(this->index() + static_cast<index_type>(1));
        return *this;
    }

    /// Check wether the link is valid to use.
    DETRAY_HOST_DEVICE
    constexpr bool is_invalid() const noexcept {
        return encoder::template is_invalid<id_mask, index_mask>(m_value);
    }

    /// Check wether the type id is invalid.
    DETRAY_HOST_DEVICE
    constexpr bool is_invalid_id() const noexcept {
        return encoder::template is_invalid<id_mask>(m_value);
    }

    /// Check wether the index is invalid.
    DETRAY_HOST_DEVICE
    constexpr bool is_invalid_index() const noexcept {
        return encoder::template is_invalid<index_mask>(m_value);
    }

    /// Print the index
    DETRAY_HOST
    friend std::ostream& operator<<(std::ostream& os, const typed_index ti) {
        if (ti.is_invalid()) {
            return (os << "undefined");
        }

        constexpr std::array names{"id = ", "index = "};
        const std::array levels{static_cast<value_t>(ti.id()),
                                static_cast<value_t>(ti.index())};

        bool writeSeparator = false;
        for (auto i = 0u; i < 2u; ++i) {
            if (writeSeparator) {
                os << " | ";
            }
            os << names[i] << levels[i];
            writeSeparator = true;
        }
        return os;
    }

    private:
    /// The encoded value. Default: All bits set to 1 (invalid)
    value_t m_value = ~static_cast<value_t>(0);
};

/// Custom get function for the typed_index struct. Get the type.
template <std::size_t idx, typename index_type, std::size_t index_size>
DETRAY_HOST_DEVICE constexpr decltype(auto) get(
    const multi_index<index_type, index_size>& index) noexcept {
    return index.indices[idx];
}

/// Custom get function for the typed_index struct. Get the type.
template <std::size_t idx, typename index_type, std::size_t index_size>
DETRAY_HOST_DEVICE constexpr decltype(auto) get(
    multi_index<index_type, index_size>& index) noexcept {
    return index.indices[idx];
}

/// Custom get function for the typed_index struct. Get the type.
template <std::size_t ID, typename id_type, typename index_type>
requires(ID == 0) DETRAY_HOST_DEVICE constexpr decltype(auto)
    get(const typed_index<id_type, index_type>& index) noexcept {
    return index.id();
}

/// Custom get function for the typed_index struct. Get the index.
template <std::size_t ID, typename id_type, typename index_type>
requires(ID == 1) DETRAY_HOST_DEVICE constexpr decltype(auto)
    get(const typed_index<id_type, index_type>& index) noexcept {
    return index.index();
}

/// Overload to check for an invalid typed index link @param ti
template <typename id_t, typename index_t, typename value_t, value_t id_mask,
          value_t index_mask>
DETRAY_HOST_DEVICE constexpr bool is_invalid_value(
    const typed_index<id_t, index_t, value_t, id_mask, index_mask>&
        ti) noexcept {
    return ti.is_invalid();
}

}  // namespace detail

}  // namespace detray
