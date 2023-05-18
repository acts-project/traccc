/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/fitting/detail/chi2_cdf.hpp"

// GTest include(s).
#include <gtest/gtest.h>

using namespace traccc;

// Test gamma function for pvalue evaulation
TEST(pvalue, compare_with_ROOT) {

    // zero chisquare -> pvalue = 1
    EXPECT_FLOAT_EQ(detail::chisquared_cdf_c(0., 0.1), 1.);
    EXPECT_FLOAT_EQ(detail::chisquared_cdf_c(0., 1.), 1.);
    EXPECT_FLOAT_EQ(detail::chisquared_cdf_c(0., 10.), 1.);

    // Compare with the outputs of ROOT::Math::chi2squared_cdf_c(chi2, ndf)
    EXPECT_FLOAT_EQ(detail::chisquared_cdf_c(3., 5.), 0.69998584);
    EXPECT_FLOAT_EQ(detail::chisquared_cdf_c(17.4, 6.26), 0.0094291369);
}