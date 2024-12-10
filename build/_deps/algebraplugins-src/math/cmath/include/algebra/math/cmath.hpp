/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Impl include(s).
#include "algebra/math/impl/cmath_getter.hpp"
#include "algebra/math/impl/cmath_matrix.hpp"
#include "algebra/math/impl/cmath_operators.hpp"
#include "algebra/math/impl/cmath_transform3.hpp"
#include "algebra/math/impl/cmath_vector.hpp"
// Algorithms include(s).
#include "algebra/math/algorithms/matrix/decomposition/partial_pivot_lud.hpp"
#include "algebra/math/algorithms/matrix/determinant/cofactor.hpp"
#include "algebra/math/algorithms/matrix/determinant/hard_coded.hpp"
#include "algebra/math/algorithms/matrix/determinant/partial_pivot_lud.hpp"
#include "algebra/math/algorithms/matrix/inverse/cofactor.hpp"
#include "algebra/math/algorithms/matrix/inverse/hard_coded.hpp"
#include "algebra/math/algorithms/matrix/inverse/partial_pivot_lud.hpp"
#include "algebra/math/algorithms/utils/algorithm_finder.hpp"