/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cstddef>

#include <gtest/gtest.h>

#include <covfie/core/backend/primitive/constant.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/field.hpp>

TEST(TestAffineTransformer, AffineConstant1Dto1D)
{
    using field_t = covfie::field<covfie::backend::affine<
        covfie::backend::
            constant<covfie::vector::float1, covfie::vector::float1>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t(covfie::algebra::affine<1>(
            covfie::array::array<covfie::array::array<float, 2>, 1>{{0.f, 5.f}}
        )),
        field_t::backend_t::backend_t::configuration_t({5.f})
    ));

    field_t::view_t fv(f);

    for (float i = -10.f; i < 10.f; i += 1.f) {
        EXPECT_EQ(fv.at(i)[0], 5.f);
    }
}
