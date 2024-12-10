/** Algebra plugins, part of the ACTS project
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Project include(s)
#include "algebra/qualifiers.hpp"
#include "algebra/storage/array_operators.hpp"

// System include(s).
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>

namespace algebra::storage {

/// Vector wrapper for AoS vs interleaved SoA data. @c value_t can e.g. be a
/// SIMD vector.
template <std::size_t N, typename value_t,
          template <typename, std::size_t> class array_t>
class vector {
 public:
  // Value type is a simd vector in SoA and a scalar in AoS
  using value_type = value_t;
  /// Underlying data array type
  using array_type = array_t<value_t, N>;

  /// Default contructor sets all entries to zero.
  constexpr vector() { zero_fill(std::make_index_sequence<N>{}); }

  /// Construct from element values @param vals .
  ///
  /// In order to avoid uninitialized values, which deteriorate the performance
  /// in explicitely vectorized code, the underlying data array is filled with
  /// zeroes if too few arguments are given.
  template <typename... Values,
            std::enable_if_t<std::conjunction_v<
                                 std::is_convertible<Values, value_type>...> &&
                                 sizeof...(Values) <= N,
                             bool> = true>
  constexpr vector(Values &&... vals) : m_data{std::forward<Values>(vals)...} {
    if constexpr ((sizeof...(Values) < N) &&
                  (!std::conjunction_v<std::is_same<array_type, Values>...>)) {
      zero_fill(std::make_index_sequence<N - sizeof...(Values)>{});
    }
  }

  /// Construct from existing array storage @param vals .
  constexpr vector(const array_type &vals) : m_data{vals} {}

  /// Assignment operator from wrapped data.
  ///
  /// @param lhs wrap a copy of this data.
  constexpr const vector &operator=(const array_type &lhs) {
    m_data = lhs;
    return *this;
  }

  /// Assignment operator from @c std::initializer_list .
  ///
  /// @param list wrap an array of this data.
  constexpr vector &operator=(std::initializer_list<value_type> &list) {
    m_data = array_type(list);
    return *this;
  }

  /// Conversion operator from wrapper to underlying data array.
  /// @{
  constexpr operator array_type &() { return m_data; }
  constexpr operator const array_type &() const { return m_data; }
  /// @}

  /// Subscript operator[]
  /// @{
  constexpr decltype(auto) operator[](std::size_t i) { return m_data[i]; }
  constexpr decltype(auto) operator[](std::size_t i) const { return m_data[i]; }
  /// @}

  /// Operator*=
  ///
  /// @return Vector expression/return type according to the operation.
  constexpr decltype(auto) operator*=(value_type factor) noexcept {
    return m_data *= factor;
  }

  /// Equality operators
  /// @{
  template <std::size_t M, typename o_value_t,
            template <typename, std::size_t> class o_array_t,
            template <typename, std::size_t> class p_array_t>
  friend constexpr bool operator==(
      const vector<M, o_value_t, o_array_t> &,
      const vector<M, o_value_t, p_array_t> &) noexcept;

  template <std::size_t M, typename o_value_t,
            template <typename, std::size_t> class o_array_t,
            template <typename, std::size_t> class p_array_t, bool>
  friend constexpr bool operator==(const vector<M, o_value_t, o_array_t> &,
                                   const p_array_t<o_value_t, M> &) noexcept;
  /// @}

  /// Inequality operator
  template <typename other_type>
  constexpr bool operator!=(const other_type &rhs) const noexcept {
    return ((*this == rhs) == false);
  }

  /// Elementwise comparison. Can result in a vector-of-masks for SoA vectors
  template <typename other_type>
  constexpr auto compare(const other_type &rhs) const noexcept {
    using result_t = decltype(m_data[0] == rhs[0]);

    array_t<result_t, N> comp;

    for (unsigned int i{0u}; i < N; ++i) {
      comp[i] = (m_data[i] == rhs[i]);
    }

    return comp;
  }

  /// Holds the data value for every vector element
  array_t<value_t, N> m_data;

 private:
  /// Sets the trailing uninitialized values to zero.
  template <std::size_t... Is>
  constexpr void zero_fill(std::index_sequence<Is...>) noexcept {
    if constexpr (sizeof...(Is) > 0) {
      //((m_data[N - sizeof...(Is) + Is] = value_t(0)), ...);
    }
  }
};

/// Friend operators
/// @{

template <std::size_t N, typename value_t,
          template <typename, std::size_t> class array_t,
          template <typename, std::size_t> class o_array_t,
          std::enable_if_t<std::is_scalar_v<value_t>, bool> = true>
constexpr bool operator==(const vector<N, value_t, array_t> &lhs,
                          const o_array_t<value_t, N> &rhs) noexcept {

  const auto comp = lhs.compare(rhs);
  bool is_full = false;

  for (unsigned int i{0u}; i < N; ++i) {
    is_full |= comp[i];
  }

  return is_full;
}

template <std::size_t N, typename value_t,
          template <typename, std::size_t> class array_t,
          template <typename, std::size_t> class o_array_t,
          std::enable_if_t<!std::is_scalar_v<value_t>, bool> = true>
constexpr bool operator==(const vector<N, value_t, array_t> &lhs,
                          const o_array_t<value_t, N> &rhs) noexcept {

  const auto comp = lhs.compare(rhs);
  bool is_full = false;

  for (unsigned int i{0u}; i < N; ++i) {
    // Ducktyping the Vc::Vector::MaskType
    is_full |= comp[i].isFull();
  }

  return is_full;
}
template <std::size_t N, typename value_t,
          template <typename, std::size_t> class array_t,
          template <typename, std::size_t> class o_array_t>
constexpr bool operator==(const vector<N, value_t, array_t> &lhs,
                          const vector<N, value_t, o_array_t> &rhs) noexcept {
  return (lhs == rhs.m_data);
}

/// @}

/// Macro declaring all instances of a specific arithmetic operator
#define DECLARE_vector_OPERATORS(OP)                                           \
  template <std::size_t N, typename value_t, typename scalar_t,                \
            template <typename, std::size_t> class array_t,                    \
            std::enable_if_t<std::is_scalar_v<scalar_t>, bool> = true>         \
  inline constexpr decltype(auto) operator OP(                                 \
      const vector<N, value_t, array_t> &lhs, scalar_t rhs) noexcept {         \
    return lhs.m_data OP static_cast<value_t>(rhs);                            \
  }                                                                            \
  template <std::size_t N, typename value_t, typename scalar_t,                \
            template <typename, std::size_t> class array_t,                    \
            std::enable_if_t<std::is_scalar_v<scalar_t>, bool> = true>         \
  inline decltype(auto) operator OP(                                           \
      scalar_t lhs, const vector<N, value_t, array_t> &rhs) noexcept {         \
    return static_cast<value_t>(lhs) OP rhs.m_data;                            \
  }                                                                            \
  template <std::size_t N, typename value_t,                                   \
            template <typename, std::size_t> class array_t>                    \
  inline constexpr decltype(auto) operator OP(                                 \
      const vector<N, value_t, array_t> &lhs,                                  \
      const vector<N, value_t, array_t> &rhs) noexcept {                       \
    return lhs.m_data OP rhs.m_data;                                           \
  }                                                                            \
  template <                                                                   \
      std::size_t N, typename value_t,                                         \
      template <typename, std::size_t> class array_t, typename other_type,     \
      std::enable_if_t<                                                        \
          std::is_object<decltype(                                             \
              std::declval<typename vector<N, value_t, array_t>::array_type>() \
                  OP std::declval<other_type>())>::value,                      \
          bool> = true>                                                        \
  inline constexpr decltype(auto) operator OP(                                 \
      const vector<N, value_t, array_t> &lhs,                                  \
      const other_type &rhs) noexcept {                                        \
    return lhs.m_data OP rhs;                                                  \
  }                                                                            \
  template <                                                                   \
      std::size_t N, typename value_t,                                         \
      template <typename, std::size_t> class array_t, typename other_type,     \
      std::enable_if_t<                                                        \
          std::is_object<decltype(                                             \
              std::declval<typename vector<N, value_t, array_t>::array_type>() \
                  OP std::declval<other_type>())>::value,                      \
          bool> = true>                                                        \
  inline constexpr decltype(auto) operator OP(                                 \
      const other_type &lhs,                                                   \
      const vector<N, value_t, array_t> &rhs) noexcept {                       \
    return lhs OP rhs.m_data;                                                  \
  }

// Implement all arithmetic operations on top of @c vector.
// clang-format off
DECLARE_vector_OPERATORS(+)
DECLARE_vector_OPERATORS(-)
DECLARE_vector_OPERATORS(*)
DECLARE_vector_OPERATORS(/)
// clang-format on

// Clean up.
#undef DECLARE_vector_OPERATORS

}  // namespace algebra::storage
