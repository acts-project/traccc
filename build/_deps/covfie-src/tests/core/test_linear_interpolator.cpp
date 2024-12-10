/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cmath>
#include <cstddef>

#include <gtest/gtest.h>

#include <covfie/core/backend/primitive/identity.hpp>
#include <covfie/core/backend/transformer/covariant_cast.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/field.hpp>

TEST(TestLinearInterpolator, Identity1D)
{
    using field_t =
        covfie::field<covfie::backend::linear<covfie::backend::covariant_cast<
            float,
            covfie::backend::identity<covfie::vector::int1>>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::backend_t::configuration_t({})
    ));
    field_t::view_t fv(f);

    for (float x = -10.f; x < 10.f; x += 0.05f) {
        EXPECT_NEAR(fv.at(x)[0], x, 0.01f);
    }
}

TEST(TestLinearInterpolator, Identity2D)
{
    using field_t =
        covfie::field<covfie::backend::linear<covfie::backend::covariant_cast<
            float,
            covfie::backend::identity<covfie::vector::int2>>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::backend_t::configuration_t({})
    ));
    field_t::view_t fv(f);

    for (float x = -5.f; x < 5.f; x += 0.05f) {
        for (float y = -5.f; y < 5.f; y += 0.05f) {
            EXPECT_NEAR(fv.at(x, y)[0], x, 0.01f);
            EXPECT_NEAR(fv.at(x, y)[1], y, 0.01f);
        }
    }
}

TEST(TestLinearInterpolator, Identity3D)
{
    using field_t =
        covfie::field<covfie::backend::linear<covfie::backend::covariant_cast<
            float,
            covfie::backend::identity<covfie::vector::int3>>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::backend_t::configuration_t({})
    ));
    field_t::view_t fv(f);

    for (float x = -2.f; x < 2.f; x += 0.2f) {
        for (float y = -2.f; y < 2.f; y += 0.2f) {
            for (float z = -2.f; z < 2.f; z += 0.2f) {
                EXPECT_NEAR(fv.at(x, y, z)[0], x, 0.01f);
                EXPECT_NEAR(fv.at(x, y, z)[1], y, 0.01f);
                EXPECT_NEAR(fv.at(x, y, z)[2], z, 0.01f);
            }
        }
    }
}

TEST(TestLinearInterpolator, Identity7D)
{
    using field_t =
        covfie::field<covfie::backend::linear<covfie::backend::covariant_cast<
            float,
            covfie::backend::identity<covfie::vector::vector_d<int, 7>>>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::backend_t::configuration_t({})
    ));
    field_t::view_t fv(f);

    for (float x = -2.f; x < 2.f; x += 0.2f) {
        for (float y = -2.f; y < 2.f; y += 0.2f) {
            for (float z = -2.f; z < 2.f; z += 0.2f) {
                EXPECT_NEAR(
                    fv.at(x, y, z, 5.4f, 2.1f, -6.6f, -0.2f)[0], x, 0.01f
                );
                EXPECT_NEAR(
                    fv.at(x, y, z, 5.4f, 2.1f, -6.6f, -0.2f)[1], y, 0.01f
                );
                EXPECT_NEAR(
                    fv.at(x, y, z, 5.4f, 2.1f, -6.6f, -0.2f)[2], z, 0.01f
                );
                EXPECT_NEAR(
                    fv.at(x, y, z, 5.4f, 2.1f, -6.6f, -0.2f)[3], 5.4f, 0.01f
                );
                EXPECT_NEAR(
                    fv.at(x, y, z, 5.4f, 2.1f, -6.6f, -0.2f)[4], 2.1f, 0.01f
                );
                EXPECT_NEAR(
                    fv.at(x, y, z, 5.4f, 2.1f, -6.6f, -0.2f)[5], -6.6f, 0.01f
                );
                EXPECT_NEAR(
                    fv.at(x, y, z, 5.4f, 2.1f, -6.6f, -0.2f)[6], -0.2f, 0.01f
                );
            }
        }
    }
}
