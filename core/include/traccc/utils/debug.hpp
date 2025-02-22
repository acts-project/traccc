/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <algebra/concepts.hpp>
#include <algebra/type_traits.hpp>

#include "traccc/definitions/qualifiers.hpp"

namespace traccc {
template <::algebra::concepts::matrix T>
TRACCC_HOST_DEVICE bool matrix_is_finite(const T& mat) {
    for (std::size_t i = 0; i < ::algebra::traits::columns<T>; ++i) {
        for (std::size_t j = 0; j < ::algebra::traits::rows<T>; ++j) {
            if (!std::isfinite(getter::element(mat, i, j))) {
                return false;
            }
        }
    }
    return true;
}
}  // namespace traccc
//
//
