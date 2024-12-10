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
#include <iterator>
#include <type_traits>

namespace detray::ranges {

/// @brief Range factory that produces a sequence of values.
///
/// @see https://en.cppreference.com/w/cpp/ranges/iota_view
///
/// @tparam incr_t the incrementable type that makes up the sequence
///
/// @note If given single value, does not do infinite iteration, but only jumps
///       to next value.
/// @note Is not fit for lazy evaluation.
template <std::incrementable incr_t>
class iota_view : public detray::ranges::view_interface<iota_view<incr_t>> {

    private:
    /// @brief Nested iterator to generate a range of values on demand.
    struct iterator {

        using difference_type = std::ptrdiff_t;
        using value_type = incr_t;
        using pointer = incr_t *;
        using reference = incr_t &;
        using iterator_category = detray::ranges::bidirectional_iterator_tag;

        constexpr iterator() requires std::default_initializable<incr_t> =
            default;

        /// Parametrized Constructor
        DETRAY_HOST_DEVICE constexpr explicit iterator(incr_t i) : m_i{i} {}

        /// Increment the index
        /// @{
        DETRAY_HOST_DEVICE
        constexpr auto operator++() -> iterator & {
            ++m_i;
            return *this;
        }

        DETRAY_HOST_DEVICE constexpr auto operator++(int) -> iterator {
            auto tmp(*this);
            ++(*this);
            return tmp;
        }
        DETRAY_HOST_DEVICE
        constexpr auto operator--() -> iterator & {
            --m_i;
            return *this;
        }

        DETRAY_HOST_DEVICE constexpr auto operator--(int) -> iterator {
            auto tmp(*this);
            --(*this);
            return tmp;
        }
        /// @}

        /// @returns the current value in the sequence - copy
        DETRAY_HOST_DEVICE
        constexpr auto operator*() const
            noexcept(std::is_nothrow_copy_constructible_v<incr_t>) -> incr_t {
            return m_i;
        }

        private:
        /// @returns true if incremetables are the same
        DETRAY_HOST_DEVICE
        friend constexpr bool operator==(const iterator &lhs,
                                         const iterator &rhs) = default;

        /// Current value of sequence
        incr_t m_i{};
    };

    /// Start and end values of the sequence
    incr_t m_start;
    incr_t m_end;

    public:
    using iterator_t = iterator;

    /// Default constructor (only works if @c imrementable_t is default
    /// constructible)
    constexpr iota_view() requires std::default_initializable<incr_t> = default;

    /// Construct from an @param interval that defines start and end values.
    template <concepts::interval interval_t>
    DETRAY_HOST_DEVICE constexpr explicit iota_view(interval_t &&interval)
        : m_start{detray::detail::get<0>(std::forward<interval_t>(interval))},
          m_end{detray::detail::get<1>(std::forward<interval_t>(interval))} {}

    /// Construct from a @param start start and @param end value.
    DETRAY_HOST_DEVICE constexpr iota_view(incr_t start, incr_t end)
        : m_start{start}, m_end{end} {}

    /// Construct from just a @param start value to represent a single value seq
    DETRAY_HOST_DEVICE
    constexpr explicit iota_view(incr_t start)
        : m_start{start}, m_end{start + 1} {}

    /// @return start position of range on container.
    DETRAY_HOST_DEVICE
    constexpr auto begin() const -> iterator_t { return iterator_t{m_start}; }

    /// @return sentinel of a sequence.
    DETRAY_HOST_DEVICE
    constexpr auto end() const -> iterator_t { return iterator_t{m_end}; }

    /// @returns the span of the sequence
    DETRAY_HOST_DEVICE
    constexpr auto size() const -> incr_t { return m_end - m_start; }
};

namespace views {

/// @brief interface type to construct a @c iota_view with CTAD
template <std::incrementable incr_t>
struct iota : public detray::ranges::iota_view<incr_t> {

    using base_type = detray::ranges::iota_view<incr_t>;

    constexpr iota() requires std::default_initializable<incr_t> = default;

    template <concepts::interval interval_t>
    DETRAY_HOST_DEVICE constexpr explicit iota(interval_t &&interval)
        : base_type(std::forward<interval_t>(interval)) {}

    DETRAY_HOST_DEVICE constexpr iota(incr_t start, incr_t end)
        : base_type(start, end) {}

    DETRAY_HOST_DEVICE constexpr explicit iota(incr_t start)
        : base_type(start) {}
};

// deduction guides

template <typename interval_t>
DETRAY_HOST_DEVICE iota(interval_t &&interval)
    ->iota<
        std::remove_cvref_t<decltype(std::get<0>(std::declval<interval_t>()))>>;

template <typename I = dindex>
DETRAY_HOST_DEVICE iota(I start, I end)->iota<I>;

template <typename I = dindex>
DETRAY_HOST_DEVICE iota(I start)->iota<I>;

}  // namespace views

}  // namespace detray::ranges
