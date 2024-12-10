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

TEST(TestCovariantCast, Identity1D)
{
    using field_t = covfie::field<covfie::backend::covariant_cast<
        float,
        covfie::backend::identity<covfie::vector::int1>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({})
    ));
    field_t::view_t fv(f);

    for (int x = -10; x < 10; x++) {
        EXPECT_EQ(fv.at(x)[0], x);
    }
}

TEST(TestCovariantCast, Identity2D)
{
    using field_t = covfie::field<covfie::backend::covariant_cast<
        float,
        covfie::backend::identity<covfie::vector::int2>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({})
    ));
    field_t::view_t fv(f);

    for (int x = -5; x < 5; x++) {
        for (int y = -5; y < 5; y++) {
            EXPECT_EQ(fv.at(x, y)[0], x);
            EXPECT_EQ(fv.at(x, y)[1], y);
        }
    }
}

TEST(TestCovariantCast, Identity3D)
{
    using field_t = covfie::field<covfie::backend::covariant_cast<
        float,
        covfie::backend::identity<covfie::vector::int3>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({})
    ));
    field_t::view_t fv(f);

    for (int x = -2; x < 2; x++) {
        for (int y = -2; y < 2; y++) {
            for (int z = -2; z < 2; z++) {
                EXPECT_EQ(fv.at(x, y, z)[0], x);
                EXPECT_EQ(fv.at(x, y, z)[1], y);
                EXPECT_EQ(fv.at(x, y, z)[2], z);
            }
        }
    }
}
