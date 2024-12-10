/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/algebra.hpp"

namespace detray {

/// Components of a bound track parameters vector.
///
enum bound_indices : unsigned int {
    // Local position on the reference surface.
    // This is intentionally named different from the position components in
    // the other data vectors, to clarify that this is defined on a surface
    // while the others are defined in free space.
    e_bound_loc0 = 0u,
    e_bound_loc1 = 1u,
    // Direction angles
    e_bound_phi = 2u,
    e_bound_theta = 3u,
    // Global inverse-momentum-like parameter, i.e. q/p or 1/p
    // The naming is inconsistent for the case of neutral track parameters where
    // the value is interpreted as 1/p not as q/p. This is intentional to avoid
    // having multiple aliases for the same element and for lack of an
    // acceptable
    // common name.
    e_bound_qoverp = 4u,
    e_bound_time = 5u,
    // Last uninitialized value contains the total number of components
    e_bound_size,
};

/// Components of a free track parameters vector.
///
/// To be used to access components by named indices instead of just numbers.
/// This must be a regular `enum` and not a scoped `enum class` to allow
/// implicit conversion to an integer. The enum value are thus visible directly
/// in `namespace Acts` and are prefixed to avoid naming collisions.
enum free_indices : unsigned int {
    // Spatial position
    // The spatial position components must be stored as one continous block.
    e_free_pos0 = 0u,
    e_free_pos1 = e_free_pos0 + 1u,
    e_free_pos2 = e_free_pos0 + 2u,
    // Time
    e_free_time = 3u,
    // (Unit) direction
    // The direction components must be stored as one continous block.
    e_free_dir0 = 4u,
    e_free_dir1 = e_free_dir0 + 1u,
    e_free_dir2 = e_free_dir0 + 2u,
    // Global inverse-momentum-like parameter, i.e. q/p or 1/p
    // See BoundIndices for further information
    e_free_qoverp = 7u,
    // Last uninitialized value contains the total number of components
    e_free_size,
};

/// Vector type for free track parametrization
template <typename algebra_t>
using free_vector =
    typename dmatrix_operator<algebra_t>::template matrix_type<e_free_size, 1>;

/// Covariance matrix type for free track parametrization
template <typename algebra_t>
using free_matrix =
    typename dmatrix_operator<algebra_t>::template matrix_type<e_free_size,
                                                               e_free_size>;

/// Vector type for bound track parametrization
template <typename algebra_t>
using bound_vector =
    typename dmatrix_operator<algebra_t>::template matrix_type<e_bound_size, 1>;

/// Covariance matrix type for bound track parametrization
template <typename algebra_t>
using bound_matrix =
    typename dmatrix_operator<algebra_t>::template matrix_type<e_bound_size,
                                                               e_bound_size>;

/// Mapping from bound to free track parameters.
template <typename algebra_t>
using bound_to_free_matrix =
    typename dmatrix_operator<algebra_t>::template matrix_type<e_free_size,
                                                               e_bound_size>;

/// Mapping from free to bound track parameters.
template <typename algebra_t>
using free_to_bound_matrix =
    typename dmatrix_operator<algebra_t>::template matrix_type<e_bound_size,
                                                               e_free_size>;

/// Mapping from free to path
template <typename algebra_t>
using free_to_path_matrix =
    typename dmatrix_operator<algebra_t>::template matrix_type<1, e_free_size>;

/// Mapping from path to free
template <typename algebra_t>
using path_to_free_matrix =
    typename dmatrix_operator<algebra_t>::template matrix_type<e_free_size, 1>;

}  // namespace detray
