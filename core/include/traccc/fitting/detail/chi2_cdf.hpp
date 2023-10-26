/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/qualifiers.hpp"
#include "traccc/fitting/detail/gamma_functions.hpp"

// Reference: ProbFuncMathCore.cxx of ROOT library
namespace traccc::detail {

// Funtions to calculate the upper incomplete gamma function from a given chi2
// and ndf
//
// @param x chi square
// @param r ndof
// @return upper incomplete gamma function (pvalue)
template <typename scalar_t>
TRACCC_HOST_DEVICE scalar_t chisquared_cdf_c(const scalar_t x,
                                             const scalar_t r) {
    double retval =
        igamc(0.5 * static_cast<double>(r), 0.5 * static_cast<double>(x));
    return static_cast<scalar_t>(retval);
}

}  // namespace traccc::detail