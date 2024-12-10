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
#include "detray/utils/ranges/ranges.hpp"
#include "detray/utils/ranges/subrange.hpp"

// System include(s)
#include <type_traits>
#include <utility>

namespace detray::ranges {

/// @brief Enumerates the elements of a range on the fly.
///
/// @tparam range_itr_t the iterator type of the enumerated range
/// @tparam incr_t a type that can be incremented in lockstep with the
///         iterator 'range_itr_t'.
///
/// @note Does not take ownership of the range it operates on. Its lifetime
/// needs to be guranteed throughout iteration or between iterations with the
/// same enumerate instance.
/// @note Is not fit for lazy evaluation.
template <std::input_iterator range_itr_t, std::incrementable incr_t>
class enumerate_view : public detray::ranges::view_interface<
                           enumerate_view<range_itr_t, incr_t>> {

    private:
    /// @brief Nested iterator to enumerate the elements of a range.
    ///
    /// The enumeration is done by incrementing an index in lockstep with a
    /// wrapped iterator of a range. Index and current iterator value are
    /// returned using structured binding.
    struct iterator {

        using itr_value_t =
            typename std::iterator_traits<range_itr_t>::value_type;
        using itr_ref_t = typename std::iterator_traits<range_itr_t>::reference;
        using itr_ptr_t = typename std::iterator_traits<range_itr_t>::pointer;

        using difference_type =
            typename std::iterator_traits<range_itr_t>::difference_type;
        using value_type = std::pair<incr_t, itr_ref_t>;
        using pointer = value_type *;
        using reference = const value_type &;
        using iterator_category =
            typename std::iterator_traits<range_itr_t>::iterator_category;

        constexpr iterator() requires std::default_initializable<range_itr_t> =
            default;

        DETRAY_HOST_DEVICE
        iterator(range_itr_t iter, incr_t offset = 0)
            : m_iter(iter), m_i{offset}, m_offset{offset} {};

        /// Increment iterator and index in lockstep
        /// @{
        DETRAY_HOST_DEVICE constexpr auto operator++() -> iterator & {
            ++m_i;
            ++m_iter;
            return *this;
        }

        DETRAY_HOST_DEVICE constexpr auto operator++(int) -> iterator {
            auto tmp(*this);
            ++(*this);
            return tmp;
        }
        /// @}

        /// Decrement iterator and index in lockstep
        /// @{
        DETRAY_HOST_DEVICE constexpr auto operator--()
            -> iterator &requires std::bidirectional_iterator<range_itr_t> {
            --m_i;
            --m_iter;
            return *this;
        }

        DETRAY_HOST_DEVICE constexpr auto operator--(int) -> iterator
            requires std::bidirectional_iterator<range_itr_t> {
            auto tmp(*this);
            ++(*this);
            return tmp;
        }
        /// @}

        /// @returns iterator and index together
        DETRAY_HOST_DEVICE constexpr auto operator*() const {
            return std::pair<incr_t, itr_ref_t>(m_i, *m_iter);
        }

        /// @returns advance this iterator state by @param j.
        DETRAY_HOST_DEVICE constexpr auto operator+=(const difference_type j)
            -> iterator &requires std::random_access_iterator<range_itr_t> {
            m_iter += j;
            m_i += static_cast<incr_t>(j);
            return *this;
        }

        /// @returns advance this iterator state by @param j.
        DETRAY_HOST_DEVICE constexpr auto operator-=(const difference_type j)
            -> iterator &requires std::random_access_iterator<range_itr_t> {
            m_iter -= j;
            m_i -= static_cast<incr_t>(j);
            return *this;
        }

        /// @returns the value and index at a given position - const
        DETRAY_HOST_DEVICE constexpr auto operator[](const difference_type i)
            const requires std::random_access_iterator<range_itr_t> {
            // check narrowing
            const incr_t index{m_offset + static_cast<incr_t>(i)};
            return std::pair<incr_t, itr_ref_t>(index, m_iter[i]);
        }

        private:
        /// @returns true if we reach end of sequence
        DETRAY_HOST_DEVICE
        friend constexpr auto operator==(const iterator &lhs,
                                         const iterator &rhs) -> bool {
            return (lhs.m_iter == rhs.m_iter);
        }

        /// @returns true if the wrapped iterators are not the same.
        DETRAY_HOST_DEVICE friend constexpr auto operator!=(const iterator &lhs,
                                                            const iterator &rhs)
            -> bool {
            return (lhs.m_iter != rhs.m_iter);
        }

        /// @returns true if we reach end of sequence
        DETRAY_HOST_DEVICE
        friend constexpr auto operator<=>(const iterator &lhs,
                                          const iterator &rhs) requires detray::
            ranges::random_access_iterator<range_itr_t> {
#if defined(__apple_build_version__)
            const auto l{lhs.m_iter};
            const auto r{rhs.m_iter};
            if (l < r || (l == r && l < r)) {
                return std::strong_ordering::less;
            }
            if (l > r || (l == r && l > r)) {
                return std::strong_ordering::greater;
            }
            return std::strong_ordering::equivalent;
#else
            return (lhs.m_iter <=> rhs.m_iter);
#endif
        }

        /// @returns an iterator and index, position advanced by @param j.
        DETRAY_HOST_DEVICE friend constexpr auto operator+(
            const iterator &itr, const difference_type j) -> iterator
            requires std::random_access_iterator<range_itr_t> {
            return {itr.m_iter + j, itr.m_i + static_cast<incr_t>(j)};
        }

        /// @returns an iterator and index, position advanced by @param j.
        DETRAY_HOST_DEVICE friend constexpr auto operator+(
            const difference_type j, const iterator &itr) -> iterator
            requires std::random_access_iterator<range_itr_t> {
            return itr + j;
        }

        /// @returns an iterator and index, position advanced by @param j.
        DETRAY_HOST_DEVICE friend constexpr auto operator-(
            const iterator &itr, const difference_type j) -> iterator
            requires std::random_access_iterator<range_itr_t> {
            return itr + (-j);
        }

        /// @returns the positional difference between two iterators
        /// (independent from their enumeration of the range values)
        DETRAY_HOST_DEVICE friend constexpr auto operator-(const iterator &lhs,
                                                           const iterator &rhs)
            -> difference_type
            requires std::random_access_iterator<range_itr_t> {
            return lhs.m_iter - rhs.m_iter;
        }

        range_itr_t m_iter{};
        incr_t m_i{0};
        incr_t m_offset{0};
    };

    iterator m_begin{};
    iterator m_end{};

    public:
    using iterator_t = iterator;

    /// Default constructor (only works if @c imrementable_t is default
    /// constructible)
    constexpr enumerate_view() = default;

    /// Construct from a @param range that will be enumerated beginning at 0
    template <detray::ranges::range range_t>
    DETRAY_HOST_DEVICE constexpr explicit enumerate_view(range_t &&rng)
        : m_begin{detray::ranges::begin(std::forward<range_t>(rng))},
          m_end{detray::ranges::end(std::forward<range_t>(rng)),
                static_cast<incr_t>(rng.size())} {}

    /// Construct from a @param range that will be enumerated beginning at
    /// @param start.
    template <detray::ranges::range range_t>
    DETRAY_HOST_DEVICE constexpr enumerate_view(range_t &&rng, incr_t start)
        : m_begin{detray::ranges::begin(std::forward<range_t>(rng)), start},
          m_end{detray::ranges::end(std::forward<range_t>(rng)),
                start + static_cast<incr_t>(rng.size())} {}

    /// @return start position of range on container.
    DETRAY_HOST_DEVICE
    constexpr auto begin() -> iterator { return m_begin; }

    /// @return start position of range on container - const
    DETRAY_HOST_DEVICE
    constexpr auto begin() const -> iterator { return m_begin; }

    /// @return sentinel of a sequence.
    DETRAY_HOST_DEVICE
    constexpr auto end() const -> iterator { return m_end; }
};

namespace views {

template <std::input_iterator range_itr_t, std::incrementable incr_t = dindex>
requires std::convertible_to<
    typename std::iterator_traits<range_itr_t>::difference_type,
    incr_t> struct enumerate : public enumerate_view<range_itr_t, incr_t> {

    using base_type = enumerate_view<range_itr_t, incr_t>;

    constexpr enumerate() = default;

    template <detray::ranges::range range_t>
    DETRAY_HOST_DEVICE constexpr explicit enumerate(range_t &&rng)
        : base_type(std::forward<range_t>(rng)) {}

    template <detray::ranges::range range_t>
    DETRAY_HOST_DEVICE constexpr enumerate(range_t &&rng, incr_t start)
        : base_type(std::forward<range_t>(rng), start) {}

    template <detray::ranges::range deduced_range_t, typename volume_t,
              typename = typename std::remove_reference_t<volume_t>::volume_def>
    DETRAY_HOST_DEVICE enumerate(deduced_range_t &&range, const volume_t &vol)
        : enumerate(detray::ranges::subrange(
                        std::forward<deduced_range_t>(range), vol),
                    detray::detail::get<0>(vol.full_range())) {}
};

// deduction guides

template <detray::ranges::range range_t>
DETRAY_HOST_DEVICE enumerate(range_t &&rng)
    ->enumerate<detray::ranges::const_iterator_t<range_t>, dindex>;

template <detray::ranges::range range_t, typename volume_t,
          typename = typename std::remove_reference_t<volume_t>::volume_def>
DETRAY_HOST_DEVICE enumerate(range_t &&range, const volume_t &vol)
    ->enumerate<detray::ranges::const_iterator_t<range_t>, dindex>;

template <detray::ranges::range range_t>
DETRAY_HOST_DEVICE enumerate(range_t &&rng, dindex start)
    ->enumerate<detray::ranges::const_iterator_t<range_t>, dindex>;

}  // namespace views

}  // namespace detray::ranges
