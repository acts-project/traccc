/** Algebra plugins, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Vc include(s).
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif  // MSVC
#include <Vc/Vc>
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // MSVC

// System include(s).
#include <cstddef>

namespace algebra::vc {

/** Wrapper structure around the Vc::SimdArray class, that allows correct
 * initialization from std::initializer_list with a different length than the
 * size of the array.
 *
 * @tparam scalar_t scalar type used in the SimdArray
 * @todo replace by @c algebra::storage::vector
 */
template <typename scalar_t>
struct array4 {

  /// The array type used internally
  using array_type = Vc::SimdArray<scalar_t, 4>;

  // Instance of the array that is being wrapped
  array_type _array;

  /** Default constructor
   */
  array4() : _array() {}

  /** Copy constructor
   *
   * @param array wrapper object to copy the data from
   */
  array4(const array4 &array) : _array(array._array) {}

  /** Move constructor
   *
   * @param array wrapper object to copy the data from
   */
  array4(array4 &&array) : _array(std::move(array._array)) {}

  /** Initialization from a single value. The Vc type broadcasts this into
   *  vector data.
   *
   * @param value The value type to construct an array from
   */
  array4(scalar_t value) : _array(value) {}

  /** Parametrized constructor that takes a number of values and makes sure that
   *  also all remaining values in the array are properly initialized.
   *
   * @param v1 value 1. dimension
   * @param v2 value 2. dimension
   * @param v3 value 3. dimension
   * @param v4 optional value 4. dimension. if not passed, set to 0.0
   *
   */
  array4(scalar_t v1, scalar_t v2, scalar_t v3,
         scalar_t v4 = static_cast<scalar_t>(0.0))
      : _array{v1, v2, v3, v4} {}

  /** Constructor from wrapped class. Used for conversions into wrapped type.
   *
   * @param base data to be wrapped
   */
  array4(array_type base) : _array(base) {}

  /** Generic assignment operator
   *
   * @param lhs wrap a copy of this data
   */
  template <typename other_type>
  inline const array4 &operator=(const other_type &lhs) {
    _array = lhs;
    return *this;
  }

  /** Assignment operator from another wrapper
   *
   * @param other wrap a copy of this wrapped data
   */
  inline array4 &operator=(const array4 &other) {
    _array = other._array;
    return *this;
  }

  /** Assignment operator from std::initializer_list
   *
   * @param list wrap an array of this data
   */
  inline array4 &operator=(std::initializer_list<scalar_t> &list) {
    _array = array_type(list);
    return *this;
  }

  /** Conversion operator from wrapper to SimdArray.
   */
  operator array_type &() { return _array; }
  operator const array_type &() const { return _array; }

  /** Operator[] overload from SimdArray for simd array wrapper.
   *
   * @return Value at given index
   */
  inline auto operator[](std::size_t i) { return _array[i]; }
  inline auto operator[](std::size_t i) const { return _array[i]; }

  /** Operator*= overload from SimdArray for simd array wrapper.
   *
   * @return Vector expression/ return type according to the operation
   */
  inline auto operator*=(scalar_t factor) { return _array *= factor; }

  /// Equality operator
  inline bool operator==(const array4 &rhs) const {
    return ((_array[0] == rhs._array[0]) && (_array[1] == rhs._array[1]) &&
            (_array[2] == rhs._array[2]) && (_array[3] == rhs._array[3]));
  }
  /// Equality operator
  template <typename other_type>
  inline bool operator==(const other_type &rhs) const {
    return ((_array[0] == rhs[0]) && (_array[1] == rhs[1]) &&
            (_array[2] == rhs[2]) && (_array[3] == rhs[3]));
  }
  /// Inequality operator
  template <typename other_type>
  inline bool operator!=(const other_type &rhs) const {
    return (this->operator==(rhs) == false);
  }
};

/// Macro declaring all instances of a specific operator
#define DECLARE_ARRAY4_OPERATORS(OP)                                      \
  template <typename scalar_t>                                            \
  inline auto operator OP(const array4<scalar_t> &lhs, double rhs) {      \
    return (lhs._array OP static_cast<scalar_t>(rhs));                    \
  }                                                                       \
  template <typename scalar_t>                                            \
  inline auto operator OP(double lhs, const array4<scalar_t> &rhs) {      \
    return (static_cast<scalar_t>(lhs) OP rhs._array);                    \
  }                                                                       \
  template <typename scalar_t>                                            \
  inline auto operator OP(const array4<scalar_t> &lhs,                    \
                          const array4<scalar_t> &rhs) {                  \
    return (lhs._array OP rhs._array);                                    \
  }                                                                       \
  template <typename scalar_t, typename other_type,                       \
            std::enable_if_t<                                             \
                std::is_object<decltype(                                  \
                    std::declval<typename array4<scalar_t>::array_type>() \
                        OP std::declval<other_type>())>::value,           \
                bool> = true>                                             \
  inline auto operator OP(const array4<scalar_t> &lhs,                    \
                          const other_type &rhs) {                        \
    return (lhs._array OP rhs);                                           \
  }                                                                       \
  template <typename scalar_t, typename other_type,                       \
            std::enable_if_t<                                             \
                std::is_object<decltype(                                  \
                    std::declval<typename array4<scalar_t>::array_type>() \
                        OP std::declval<other_type>())>::value,           \
                bool> = true>                                             \
  inline auto operator OP(const other_type &lhs,                          \
                          const array4<scalar_t> &rhs) {                  \
    return (lhs OP rhs._array);                                           \
  }

// Implement all arithmetic operations on top of @c array4.
DECLARE_ARRAY4_OPERATORS(+)
DECLARE_ARRAY4_OPERATORS(-)
DECLARE_ARRAY4_OPERATORS(*)
DECLARE_ARRAY4_OPERATORS(/)

// Clean up.
#undef DECLARE_ARRAY4_OPERATORS

}  // namespace algebra::vc
