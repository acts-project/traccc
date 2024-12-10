/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/definitions/pdg_particle.hpp"

// GTest include(s)
#include <gtest/gtest.h>

using namespace detray;

/// This tests the functionality of a pdg particle
GTEST_TEST(detray_core, pdg_particle) {

    constexpr auto mu = muon<float>();
    constexpr auto ep = positron<float>();

    ASSERT_EQ(ep.pdg_num(), -11);
    ASSERT_FLOAT_EQ(ep.charge(), 1.f);
    ASSERT_FLOAT_EQ(ep.mass(), 0.51099895069f * unit<float>::MeV);
    ASSERT_FLOAT_EQ(ep.mass(), electron<float>().mass());

    ASSERT_EQ(mu.pdg_num(), 13);
    ASSERT_FLOAT_EQ(mu.charge(), -1.f);
    ASSERT_FLOAT_EQ(mu.mass(), 105.6583755f * unit<float>::MeV);
    ASSERT_FLOAT_EQ(mu.mass(), antimuon<float>().mass());

    ASSERT_FLOAT_EQ(pion_plus<float>().mass(), pion_minus<float>().mass());
}
