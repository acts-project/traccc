/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/utils/concepts.hpp"
#include "detray/utils/ranges/ranges.hpp"

// System include(s)
#include <type_traits>

namespace detray::ranges {

/// @brief Implements a subrange by providing start and end iterators on
/// another range.
///
/// @see https://en.cppreference.com/w/cpp/ranges/subrange
///
/// @tparam range_t the iterable which to constrain to a subrange.
template <detray::ranges::range range_t>
class subrange : public detray::ranges::view_interface<subrange<range_t>> {

    public:
    using iterator_t = typename detray::ranges::iterator_t<range_t>;
    using const_iterator_t = typename detray::ranges::const_iterator_t<range_t>;
    using difference_t = typename detray::ranges::range_difference_t<range_t>;

    /// Default constructor
    constexpr subrange() = default;

    /// Construct from an @param start and @param end iterator pair.
    DETRAY_HOST_DEVICE constexpr subrange(iterator_t start, iterator_t end)
        : m_begin{start}, m_end{end} {}

    /// Construct from a @param range.
    template <detray::ranges::range deduced_range_t>
    DETRAY_HOST_DEVICE constexpr explicit subrange(deduced_range_t &&range)
        : m_begin{detray::ranges::begin(std::forward<deduced_range_t>(range))},
          m_end{detray::ranges::end(std::forward<deduced_range_t>(range))} {}

    /// Construct from a @param range and starting position @param pos. Used
    /// as an overload when only a single position is needed.
    template <detray::ranges::range deduced_range_t, typename index_t>
    requires std::is_convertible_v<
        index_t, detray::ranges::range_difference_t<deduced_range_t>>
        DETRAY_HOST_DEVICE constexpr subrange(deduced_range_t &&range,
                                              index_t pos)
        : m_begin{detray::ranges::next(
              detray::ranges::begin(std::forward<deduced_range_t>(range)),
              static_cast<difference_t>(pos))},
          m_end{detray::ranges::next(m_begin)} {}

    /// Construct from a @param range and an index range @param pos.
    template <detray::ranges::range deduced_range_t,
              concepts::interval index_range_t>
    DETRAY_HOST_DEVICE constexpr subrange(deduced_range_t &&range,
                                          index_range_t &&pos)
        : m_begin{detray::ranges::next(
              detray::ranges::begin(std::forward<deduced_range_t>(range)),
              static_cast<difference_t>(
                  detray::detail::get<0>(std::forward<index_range_t>(pos))))},
          m_end{detray::ranges::next(
              detray::ranges::begin(std::forward<deduced_range_t>(range)),
              static_cast<difference_t>(
                  detray::detail::get<1>(std::forward<index_range_t>(pos))))} {}

    /// @return start position of range.
    DETRAY_HOST_DEVICE
    constexpr auto begin() -> iterator_t { return m_begin; }

    /// @return sentinel of the range.
    DETRAY_HOST_DEVICE
    constexpr auto end() -> iterator_t { return m_end; }

    /// @return start position of the range - const
    DETRAY_HOST_DEVICE
    constexpr auto begin() const -> const_iterator_t { return m_begin; }

    /// @return sentinel of the range.
    DETRAY_HOST_DEVICE
    constexpr auto end() const -> const_iterator_t { return m_end; }

    /// Equality operator
    ///
    /// @param rhs the subrange to compare with
    ///
    /// @returns whether the two subranges are equal
    DETRAY_HOST_DEVICE
    constexpr auto operator==(const subrange &rhs) const -> bool {
        return m_begin == rhs.m_begin && m_end == rhs.m_end;
    }

    private:
    /// Start and end position of the subrange
    iterator_t m_begin;
    iterator_t m_end;
};

template <detray::ranges::range deduced_range_t>
DETRAY_HOST_DEVICE subrange(deduced_range_t &&range)->subrange<deduced_range_t>;

template <detray::ranges::range deduced_range_t, typename index_t>
requires std::convertible_to<
    index_t, detray::ranges::range_difference_t<deduced_range_t>>
    DETRAY_HOST_DEVICE subrange(deduced_range_t &&range, index_t pos)
        ->subrange<deduced_range_t>;

template <detray::ranges::range deduced_range_t,
          concepts::interval index_range_t>
DETRAY_HOST_DEVICE subrange(deduced_range_t &&range, index_range_t &&pos)
    ->subrange<deduced_range_t>;

/// @see https://en.cppreference.com/w/cpp/ranges/borrowed_iterator_t
template <detray::ranges::range R>
using borrowed_subrange_t =
    std::conditional_t<borrowed_range<R>, detray::ranges::subrange<R>,
                       dangling>;

}  // namespace detray::ranges
