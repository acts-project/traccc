/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/fitting/detail/gamma_functions.hpp"

// GTest include(s).
#include <gtest/gtest.h>

#include <iostream>

using namespace traccc;

// Test gamma function for pvalue evaulation
TEST(gamma_funtion, pvalue) {

    std::cout << detail::chisquared_cdf_c(0, 0) << std::endl;
}