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

// System include(s)
#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

namespace detray::ranges {

/// @brief Range adaptor that indexes the elements of a range using another
///        index range.
///
/// @tparam range_itr_t the iterator type of the enumerated range
/// @tparam sequence_itr_t range of indices.
///
/// @note Does not take ownership of the range it operates on. Its lifetime
/// needs to be guranteed throughout iteration or between iterations with the
/// same enumerate instance.
/// @note Is not fit for lazy evaluation.
template <std::input_iterator range_itr_t, std::input_iterator sequence_itr_t>
class pick_view : public detray::ranges::view_interface<
                      pick_view<range_itr_t, sequence_itr_t>> {

    private:
    using index_t = typename std::iterator_traits<sequence_itr_t>::value_type;
    using value_t = typename std::iterator_traits<range_itr_t>::value_type;
    using difference_t =
        typename std::iterator_traits<range_itr_t>::difference_type;

    /// @brief Nested iterator to randomly index the elements of a range.
    ///
    /// The indices by which to reference the range are obtained by a dedicated
    /// index range
    ///
    /// @todo Add Comparability to fulfill random access iterator traits once
    ///       needed.
    struct iterator {

        using itr_value_t =
            typename std::iterator_traits<range_itr_t>::value_type;
        using itr_ref_t = typename std::iterator_traits<range_itr_t>::reference;
        using itr_ptr_t = typename std::iterator_traits<range_itr_t>::pointer;

        using difference_type =
            typename std::iterator_traits<range_itr_t>::difference_type;
        using value_type =
            std::pair<typename std::iterator_traits<sequence_itr_t>::value_type,
                      itr_ref_t>;
        using pointer = value_type *;
        using reference = const value_type &;
        using iterator_category =
            typename std::iterator_traits<sequence_itr_t>::iterator_category;

        static_assert(std::is_convertible_v<index_t, difference_type>,
                      "Given sequence cannot be "
                      "used to index elements of range.");

        /// Default constructor required by LegacyIterator trait
        constexpr iterator() = default;

        /// Construct from range and sequence range
        DETRAY_HOST_DEVICE
        constexpr iterator(range_itr_t rng_itr, range_itr_t rng_begin,
                           sequence_itr_t sq_itr, sequence_itr_t sq_end)
            : m_range_iter{rng_itr},
              m_range_begin{rng_begin},
              m_seq_iter{sq_itr},
              m_seq_end{sq_end} {}

        /// Increment iterator and index in lockstep
        /// @{
        DETRAY_HOST_DEVICE constexpr auto operator++() -> iterator & {
            ++m_seq_iter;
            if (m_seq_iter != m_seq_end) {
                m_range_iter =
                    m_range_begin + static_cast<difference_type>(*m_seq_iter);
            }
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
            m_range_iter =
                m_range_begin + static_cast<difference_type>(*(--m_seq_iter));
            return *this;
        }

        DETRAY_HOST_DEVICE constexpr auto operator--(int) -> iterator
            requires std::bidirectional_iterator<range_itr_t> {
            auto tmp(*this);
            --(*this);
            return tmp;
        }

        /// @returns iterator and index together
        DETRAY_HOST_DEVICE auto operator*() const {
            return value_type(*m_seq_iter, *m_range_iter);
        }

        /// @returns advance this iterator state by @param j.
        DETRAY_HOST_DEVICE constexpr auto operator+=(const difference_type j)
            -> iterator &requires std::random_access_iterator<range_itr_t> {
            detray::ranges::advance(m_seq_iter, j);
            m_range_begin += static_cast<difference_type>(*m_seq_iter);
            return *this;
        }

        /// @returns advance this iterator state by @param j.
        DETRAY_HOST_DEVICE constexpr auto operator-=(const difference_type j)
            -> iterator &requires std::random_access_iterator<range_itr_t>
                &&std::random_access_iterator<sequence_itr_t> {
            return *this += -j;
        }

        /// @returns the value and index at a given position - const
        DETRAY_HOST_DEVICE constexpr auto operator[](const difference_type i)
            const requires std::random_access_iterator<range_itr_t>
                &&std::random_access_iterator<sequence_itr_t> {
            const auto index{m_seq_iter[i]};
            return value_type(index, m_range_begin[index]);
        }

        private:
        /// @returns true if we reach end of sequence
        DETRAY_HOST_DEVICE
        friend constexpr auto operator==(const iterator &lhs,
                                         const iterator &rhs) -> bool {
            return (lhs.m_seq_iter == rhs.m_seq_iter);
        }

        /// @returns true if we reach end of sequence
        DETRAY_HOST_DEVICE
        friend constexpr auto operator<=>(const iterator &lhs,
                                          const iterator &rhs) requires detray::
            ranges::random_access_iterator<sequence_itr_t> {
#if defined(__apple_build_version__)
            const auto l{lhs.m_seq_iter};
            const auto r{rhs.m_seq_iter};
            if (l < r || (l == r && l < r)) {
                return std::strong_ordering::less;
            }
            if (l > r || (l == r && l > r)) {
                return std::strong_ordering::greater;
            }
            return std::strong_ordering::equivalent;
#else
            return (lhs.m_seq_iter <=> rhs.m_seq_iter);
#endif
        }

        /// @returns an iterator and index position advanced by @param j.
        DETRAY_HOST_DEVICE friend constexpr auto operator+(
            const iterator &itr, const difference_type j) -> iterator
            requires std::random_access_iterator<range_itr_t> {
            auto seq_iter = detray::ranges::next(itr.m_seq_iter, j);
            return {itr.m_range_begin + static_cast<difference_type>(*seq_iter),
                    itr.m_range_begin, seq_iter, itr.m_seq_end};
        }

        /// @returns an iterator and index position advanced by @param j.
        DETRAY_HOST_DEVICE friend constexpr auto operator+(
            const difference_type j, const iterator &itr) -> iterator
            requires std::random_access_iterator<range_itr_t> {
            return itr + j;
        }

        /// @returns an iterator and index position advanced by @param j.
        DETRAY_HOST_DEVICE friend constexpr auto operator-(
            const iterator &itr, const difference_type j) -> iterator
            requires std::random_access_iterator<range_itr_t> {
            return itr + (-j);
        }

        /// @returns the positional difference between two iterations
        DETRAY_HOST_DEVICE friend constexpr auto operator-(const iterator &lhs,
                                                           const iterator &rhs)
            -> difference_type requires std::random_access_iterator<range_itr_t>
                &&std::random_access_iterator<sequence_itr_t> {
            return lhs.m_seq_iter - rhs.m_seq_iter;
        }

        range_itr_t m_range_iter{};
        range_itr_t m_range_begin{};
        sequence_itr_t m_seq_iter{};
        sequence_itr_t m_seq_end{};
    };

    range_itr_t m_range_begin{};
    range_itr_t m_range_end{};
    sequence_itr_t m_seq_begin{};
    sequence_itr_t m_seq_end{};

    public:
    using iterator_t = iterator;

    /// Default constructor
    constexpr pick_view() = default;

    /// Construct from a @param range that will be enumerated beginning at 0
    template <detray::ranges::range range_t, detray::ranges::range sequence_t>
    DETRAY_HOST_DEVICE constexpr pick_view(range_t &&range, sequence_t &&seq)
        : m_range_begin{detray::ranges::begin(std::forward<range_t>(range))},
          m_range_end{detray::ranges::end(std::forward<range_t>(range))},
          m_seq_begin{detray::ranges::cbegin(std::forward<sequence_t>(seq))},
          m_seq_end{detray::ranges::cend(std::forward<sequence_t>(seq))} {}

    /// @return start position of range on container.
    DETRAY_HOST_DEVICE
    constexpr auto begin() -> iterator {
        return {m_range_begin + static_cast<difference_t>(*m_seq_begin),
                m_range_begin, m_seq_begin, m_seq_end};
    }

    /// @return start position of range on container.
    DETRAY_HOST_DEVICE
    constexpr auto begin() const -> iterator {
        return {m_range_begin + static_cast<difference_t>(*m_seq_begin),
                m_range_begin, m_seq_begin, m_seq_end};
    }

    /// @return sentinel of a sequence.
    DETRAY_HOST_DEVICE
    constexpr auto end() -> iterator {
        return {m_range_begin, m_range_begin, m_seq_end, m_seq_end};
    }

    /// @returns a pointer to the beginning of the data of the first underlying
    /// range - const
    DETRAY_HOST_DEVICE
    constexpr auto data() const -> const typename iterator_t::value_type * {
        return std::addressof(
            *(m_range_begin + static_cast<difference_t>(*m_seq_begin)));
    }

    /// @returns a pointer to the beginning of the data of the first underlying
    /// range - non-const
    DETRAY_HOST_DEVICE
    constexpr auto data() -> typename iterator_t::value_type * {
        return std::addressof(
            *(m_range_begin + static_cast<difference_t>(*m_seq_begin)));
    }

    /// @returns the number of elements in the underlying range. Simplified
    /// implementation compared to @c view_interface.
    DETRAY_HOST_DEVICE
    constexpr auto size() const noexcept {
        return detray::ranges::distance(m_seq_begin, m_seq_end);
    }

    /// @returns access to the first element value in the range, including the
    /// corresponding index.
    DETRAY_HOST_DEVICE
    constexpr auto front() noexcept {
        return std::pair<index_t, value_t &>(
            *m_seq_begin,
            *detray::ranges::next(m_range_begin,
                                  static_cast<difference_t>(*m_seq_begin)));
    }

    /// @returns access to the last element value in the range, including the
    /// corresponding index.
    DETRAY_HOST_DEVICE
    constexpr auto back() noexcept {
        index_t last_idx{*detray::ranges::next(
            m_seq_begin, static_cast<difference_t>((size() - 1u)))};
        return std::pair<index_t, value_t &>(
            last_idx, *detray::ranges::next(
                          m_range_begin, static_cast<difference_t>(last_idx)));
    }

    DETRAY_HOST_DEVICE
    constexpr auto operator[](const dindex i) const {
        index_t last_idx{
            *detray::ranges::next(m_seq_begin, static_cast<difference_t>(i))};
        return std::pair<index_t, value_t &>(
            last_idx, *detray::ranges::next(
                          m_range_begin, static_cast<difference_t>(last_idx)));
    }
};

namespace views {

template <std::input_iterator range_itr_t, std::input_iterator sequence_itr_t>
struct pick : public pick_view<range_itr_t, sequence_itr_t> {

    using base_type = pick_view<range_itr_t, sequence_itr_t>;

    pick() = default;

    template <detray::ranges::range range_t, detray::ranges::range sequence_t>
    DETRAY_HOST_DEVICE constexpr pick(range_t &&range, sequence_t &&seq)
        : base_type(std::forward<range_t>(range),
                    std::forward<sequence_t>(seq)) {}
};

// deduction guides

template <detray::ranges::range range_t, detray::ranges::range sequence_t>
pick(range_t &&range, sequence_t &&seq)
    -> pick<detray::ranges::iterator_t<range_t>,
            detray::ranges::const_iterator_t<sequence_t>>;

}  // namespace views

}  // namespace detray::ranges
