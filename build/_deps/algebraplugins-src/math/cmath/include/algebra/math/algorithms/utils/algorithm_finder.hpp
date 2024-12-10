/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include
#include <utility>

namespace algebra::cmath {

template <typename size_type, size_type... Is>
constexpr bool is_in([[maybe_unused]] size_type i,
                     std::integer_sequence<size_type, Is...>) {
  return ((i == Is) || ...);
}

template <typename size_type, size_type N, typename A, typename... As>
struct find_algorithm_or_default;

template <typename size_type, size_type N, typename A>
struct find_algorithm_or_default<size_type, N, A> {

  using algorithm_type = A;
  static constexpr bool found = false;
};

template <typename size_type, size_type N, typename Default, typename A,
          typename... As>
struct find_algorithm_or_default<size_type, N, Default, A, As...> {

 private:
  using next = find_algorithm_or_default<size_type, N, Default, As...>;

 public:
  using algorithm_type =
      typename std::conditional_t<is_in(N, typename A::_dims{}), A,
                                  typename next::algorithm_type>;

  static constexpr bool found = is_in(N, typename A::_dims{}) || next::found;
};

template <typename size_type, size_type N, typename Default, typename... Ts>
struct find_algorithm {
 private:
  using helper =
      find_algorithm_or_default<size_type, N, Default, Default, Ts...>;

 public:
  using algorithm_type = typename helper::algorithm_type;
  static constexpr bool found = helper::found;
};

}  // namespace algebra::cmath
