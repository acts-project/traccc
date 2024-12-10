/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/tracks/bound_track_parameters.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// Google Test include(s)
#include <gtest/gtest.h>

using namespace detray;

using algebra_t = test::algebra;
using point2 = test::point2;
using vector3 = test::vector3;
using point3 = test::point3;
using matrix_operator = test::matrix_operator;

constexpr scalar tol{1e-5f};

GTEST_TEST(detray_tracks, bound_track_parameters) {

    constexpr scalar charge = -1.f;

    /// Declare track parameters
    dindex sf_idx1 = 0u;
    point2 bound_local{1.f, 2.f};
    scalar phi{0.1f};
    scalar theta{0.2f};
    scalar qop{-0.1f};
    scalar t{0.1f};

    // first track
    bound_parameters_vector<algebra_t> bound_vec1{bound_local, phi, theta, qop,
                                                  t};

    typename bound_track_parameters<algebra_t>::covariance_type bound_cov1 =
        matrix_operator().template zero<e_bound_size, e_bound_size>();

    bound_track_parameters<algebra_t> bound_param1(
        geometry::barcode{}.set_index(sf_idx1), bound_vec1, bound_cov1);
    EXPECT_NEAR(bound_param1.pT(charge),
                1.f / std::abs(bound_vec1.qop()) * std::sin(bound_vec1.theta()),
                tol);
    EXPECT_NEAR(bound_param1.qopT(), -1.f / bound_param1.pT(charge), tol);
    EXPECT_NEAR(bound_param1.pz(charge),
                1.f / std::abs(bound_vec1.qop()) * std::cos(bound_vec1.theta()),
                tol);
    EXPECT_NEAR(bound_param1.qopz(), -1.f / bound_param1.pz(charge), tol);

    // second track
    dindex sf_idx2 = 1u;
    bound_local = {4.f, 20.f};
    phi = 0.8f;
    theta = 1.4f;
    qop = -1.f;
    t = 0.f;

    bound_parameters_vector<algebra_t> bound_vec2{bound_local, phi, theta, qop,
                                                  t};

    typename bound_track_parameters<algebra_t>::covariance_type bound_cov2 =
        matrix_operator().template zero<e_bound_size, e_bound_size>();

    bound_track_parameters<algebra_t> bound_param2(
        geometry::barcode{}.set_index(sf_idx2), bound_vec2, bound_cov2);
    bound_track_parameters<algebra_t> bound_param3(
        geometry::barcode{}.set_index(sf_idx2), bound_vec2, bound_cov2);

    /// Check the elements

    // first track
    EXPECT_NEAR(bound_param1.bound_local()[0], bound_vec1.bound_local()[0],
                tol);
    EXPECT_NEAR(bound_param1.bound_local()[1], bound_vec1.bound_local()[1],
                tol);
    EXPECT_NEAR(bound_param1.phi(), bound_vec1.phi(), tol);
    EXPECT_NEAR(bound_param1.theta(), bound_vec1.theta(), tol);
    EXPECT_NEAR(bound_param1.qop(), bound_vec1.qop(), tol);
    EXPECT_NEAR(bound_param1.time(), bound_vec1.time(), tol);
    EXPECT_NEAR(bound_param1.mom(charge)[0],
                bound_param1.p(charge) * std::sin(bound_param1.theta()) *
                    std::cos(bound_param1.phi()),
                tol);
    EXPECT_NEAR(bound_param1.mom(charge)[1],
                bound_param1.p(charge) * std::sin(bound_param1.theta()) *
                    std::sin(bound_param1.phi()),
                tol);
    EXPECT_NEAR(bound_param1.mom(charge)[2],
                bound_param1.p(charge) * std::cos(bound_param1.theta()), tol);

    // second track
    EXPECT_NEAR(bound_param2.bound_local()[0], bound_vec2.bound_local()[0],
                tol);
    EXPECT_NEAR(bound_param2.bound_local()[1], bound_vec2.bound_local()[1],
                tol);
    EXPECT_NEAR(bound_param2.phi(), bound_vec2.phi(), tol);
    EXPECT_NEAR(bound_param2.theta(), bound_vec2.theta(), tol);
    EXPECT_NEAR(bound_param2.qop(), bound_vec2.qop(), tol);
    EXPECT_NEAR(bound_param2.time(), bound_vec2.time(), tol);
    EXPECT_NEAR(bound_param2.mom(charge)[0],
                bound_param2.p(charge) * std::sin(bound_param2.theta()) *
                    std::cos(bound_param2.phi()),
                tol);
    EXPECT_NEAR(bound_param2.mom(charge)[1],
                bound_param2.p(charge) * std::sin(bound_param2.theta()) *
                    std::sin(bound_param2.phi()),
                tol);
    EXPECT_NEAR(bound_param2.mom(charge)[2],
                bound_param2.p(charge) * std::cos(bound_param2.theta()), tol);

    EXPECT_TRUE(!(bound_param2 == bound_param1));
    EXPECT_TRUE(bound_param2 == bound_param3);

    bound_param2.set_qop(0.127f);
    EXPECT_FLOAT_EQ(static_cast<float>(bound_param2.qop()), 0.127f);
}
