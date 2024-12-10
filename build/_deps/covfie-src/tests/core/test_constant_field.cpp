/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cstddef>

#include <gtest/gtest.h>

#include <covfie/core/backend/primitive/constant.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/parameter_pack.hpp>

TEST(TestConstantField, Constant1Dto1D)
{
    using field_t = covfie::field<covfie::backend::constant<
        covfie::vector::float1,
        covfie::vector::float1>>;

    field_t f(
        covfie::make_parameter_pack(field_t::backend_t::configuration_t({5.f}))
    );

    field_t::view_t fv(f);

    for (float i = -10.f; i < 10.f; i += 1.f) {
        EXPECT_EQ(fv.at(i)[0], 5.f);
    }
}

TEST(TestConstantField, Constant1Dto3D)
{
    using field_t = covfie::field<covfie::backend::constant<
        covfie::vector::float1,
        covfie::vector::float3>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({5.f, 2.f, 8.f})
    ));

    field_t::view_t fv(f);

    for (float i = -10.f; i < 10.f; i += 1.f) {
        EXPECT_EQ(fv.at(i)[0], 5.f);
        EXPECT_EQ(fv.at(i)[1], 2.f);
        EXPECT_EQ(fv.at(i)[2], 8.f);
    }
}

TEST(TestConstantField, Constant3Dto1D)
{
    using field_t = covfie::field<covfie::backend::constant<
        covfie::vector::float3,
        covfie::vector::float1>>;

    field_t f(
        covfie::make_parameter_pack(field_t::backend_t::configuration_t({5.f}))
    );

    field_t::view_t fv(f);

    for (float x = -10.f; x < 10.f; x += 1.f) {
        for (float y = -10.f; y < 10.f; y += 1.f) {
            for (float z = -10.f; z < 10.f; z += 1.f) {
                EXPECT_EQ(fv.at(x, y, z)[0], 5.f);
            }
        }
    }
}

TEST(TestConstantField, Constant3Dto3D)
{
    using field_t = covfie::field<covfie::backend::constant<
        covfie::vector::float3,
        covfie::vector::float3>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({5.f, 2.f, 8.f})
    ));

    field_t::view_t fv(f);

    for (float x = -10.f; x < 10.f; x += 1.f) {
        for (float y = -10.f; y < 10.f; y += 1.f) {
            for (float z = -10.f; z < 10.f; z += 1.f) {
                EXPECT_EQ(fv.at(x, y, z)[0], 5.f);
                EXPECT_EQ(fv.at(x, y, z)[1], 2.f);
                EXPECT_EQ(fv.at(x, y, z)[2], 8.f);
            }
        }
    }
}
