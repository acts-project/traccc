/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem {
namespace edm {
namespace type {

/// @name Meta-types describing individual variables in an SoA container
/// @{

/// @brief Scalar variable, one for a whole container
/// @tparam TYPE The type of the scalar variable
template <typename TYPE>
struct scalar {
    using type = TYPE;
};  // struct scalar

/// @brief 1D vector variable
/// @tparam TYPE The type of the 1D vector
template <typename TYPE>
struct vector {
    using type = TYPE;
};  // struct vector

/// @brief 2D jagged vector variable
/// @tparam TYPE The type of the 2D jagged vector
template <typename TYPE>
struct jagged_vector {
    using type = TYPE;
};  // struct jagged_vector

/// @}

}  // namespace type

/// Meta type describing the "schema" of an SoA container
///
/// @tparam ...VARTYPES The variable types in the SoA container
///
template <typename... VARTYPES>
struct schema {};

}  // namespace edm
}  // namespace vecmem
