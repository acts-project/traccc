/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#if defined(ALGEBRA_array)
#include "plugins/algebra/array_definitions.hpp"
#elif defined(ALGEBRA_eigen)
#include "plugins/algebra/eigen_definitions.hpp"
#elif defined(ALGEBRA_smatrix)
#include "plugins/algebra/smatrix_definitions.hpp"
#elif defined(ALGEBRA_vc)
#include "plugins/algebra/vc_array_definitions.hpp"
#endif
