/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

namespace covfie::utility {
/**
 * @brief Utility to concatenate two type-level index sequences.
 *
 * This is the unspecialised version of the utility, which basically does
 * nothing. It needs to be matched to two actual index sequences to work.
 */
template <typename, typename>
struct concat_index_sequence {
};

/*
 * This is a specialization of `concat_index_sequence` specialized to work on
 * two index sequences, this simply takes two sequences and puts one after the
 * other in a new type.
 */
template <std::size_t... L, std::size_t... H>
struct concat_index_sequence<
    std::index_sequence<L...>,
    std::index_sequence<H...>> {
    using type = std::index_sequence<L..., H...>;
};

/**
 * @brief Utility to filter index sequences, keeping only values smaller than
 * a given pivot.
 *
 * This is the unspecialised version of the utility, which basically does
 * nothing. It needs to be matched to a pivot and an index sequence to work.
 */
template <typename, typename>
struct filter_index_sequence_lt {
};

/*
 * This is a specialization of `filter_index_sequence_lt` that works on empty
 * sequences. Obviously, this just returns an empty sequence.
 */
template <std::size_t N>
struct filter_index_sequence_lt<
    std::integral_constant<std::size_t, N>,
    std::index_sequence<>> {
    using type = std::index_sequence<>;
};

/*
 * This is a specialization of `filter_index_sequence_lt` that works on
 * non-empty sequences, which consist of a head V and a tail Vs. This works
 * recursively.
 */
template <std::size_t N, std::size_t V, std::size_t... Vs>
struct filter_index_sequence_lt<
    std::integral_constant<std::size_t, N>,
    std::index_sequence<V, Vs...>> {
    using type = typename concat_index_sequence <
                 std::conditional_t<
                     V<N, std::index_sequence<V>, std::index_sequence<>>,
                     typename filter_index_sequence_lt<
                         std::integral_constant<std::size_t, N>,
                         std::index_sequence<Vs...>>::type>::type;
};

/**
 * @brief Utility to filter index sequences, keeping only values greater than
 * or equal to a given pivot.
 *
 * This is the unspecialised version of the utility, which basically does
 * nothing. It needs to be matched to a pivot and an index sequence to work.
 */
template <typename, typename>
struct filter_index_sequence_geq {
};

/*
 * This is a specialization of `filter_index_sequence_geq` that works on empty
 * sequences. Obviously, this just returns an empty sequence.
 */
template <std::size_t N>
struct filter_index_sequence_geq<
    std::integral_constant<std::size_t, N>,
    std::index_sequence<>> {
    using type = std::index_sequence<>;
};

/*
 * This is a specialization of `filter_index_sequence_geq` that works on
 * non-empty sequences, which consist of a head V and a tail Vs. This works
 * recursively.
 */
template <std::size_t N, std::size_t V, std::size_t... Vs>
struct filter_index_sequence_geq<
    std::integral_constant<std::size_t, N>,
    std::index_sequence<V, Vs...>> {
    using type = typename concat_index_sequence<
        std::conditional_t<
            V >= N,
            std::index_sequence<V>,
            std::index_sequence<>>,
        typename filter_index_sequence_geq<
            std::integral_constant<std::size_t, N>,
            std::index_sequence<Vs...>>::type>::type;
};

/**
 * @brief Utility to sort a compile-time index sequence using a merge sort
 * algorithm.
 *
 * This is the unspecialised version of the utility, which basically does
 * nothing. It needs to be matched to a pivot and an index sequence to work.
 */
template <typename>
struct sort_index_sequence {
};

/*
 * This is a specialization of `sort_index_sequence` that works on empty
 * sequences, which obviously returns an empty sequence.
 */
template <>
struct sort_index_sequence<std::index_sequence<>> {
    using type = std::index_sequence<>;
};

/*
 * This is a specialization of `sort_index_sequence` that works on non-empty
 * sequences consisting of a head N and a tail Ns. The head N is used as a
 * pivot, and the tail is split in two parts: the lower half (with values
 * smaller than N) and the upper part (with values greater than or equal to N).
 * These are then concatenated, with the pivot N in between them.
 */
template <std::size_t N, std::size_t... Ns>
struct sort_index_sequence<std::index_sequence<N, Ns...>> {
    using type = typename concat_index_sequence<
        typename sort_index_sequence<typename filter_index_sequence_lt<
            std::integral_constant<std::size_t, N>,
            std::index_sequence<Ns...>>::type>::type,
        typename concat_index_sequence<
            std::index_sequence<N>,
            typename sort_index_sequence<typename filter_index_sequence_geq<
                std::integral_constant<std::size_t, N>,
                std::index_sequence<Ns...>>::type>::type>::type>::type;
};

/**
 * @brief Utility to check if two index sequences are permutations of each
 * other.
 *
 * This is the unspecialised version of the utility, which basically does
 * nothing. It needs to be matched to a pivot and an index sequence to work.
 */
template <typename, typename>
struct is_permutation : std::false_type {
};

/*
 * This is a specialization of `is_permutation`, which simply sorts both
 * sequences and then checks if they are equal! Simplest algorithm to check for
 * permutations in the book.
 */
template <std::size_t... Us, std::size_t... Vs>
struct is_permutation<std::index_sequence<Us...>, std::index_sequence<Vs...>>
    : std::is_same<
          typename sort_index_sequence<std::index_sequence<Us...>>::type,
          typename sort_index_sequence<std::index_sequence<Vs...>>::type> {
};
}
