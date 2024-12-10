/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cstddef>

#include <gtest/gtest.h>

#include <covfie/core/backend/primitive/identity.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/vector.hpp>

TEST(TestCanonicalLayout, Canonical1D)
{
    using field_t = covfie::field<covfie::backend::strided<
        covfie::vector::size1,
        covfie::backend::identity<covfie::vector::size1>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t{16u},
        field_t::backend_t::backend_t::configuration_t{}
    ));
    field_t::view_t fv(f);

    for (std::size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(fv.at(i)[0], i);
    }
}

TEST(TestCanonicalLayout, Canonical2D)
{
    using field_t = covfie::field<covfie::backend::strided<
        covfie::vector::size2,
        covfie::backend::identity<covfie::vector::size1>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t{4u, 4u},
        field_t::backend_t::backend_t::configuration_t{}
    ));

    field_t::view_t fv(f);

    EXPECT_EQ(fv.at(0, 0)[0], 0);
    EXPECT_EQ(fv.at(0, 3)[0], 3);
    EXPECT_EQ(fv.at(1, 1)[0], 5);
    EXPECT_EQ(fv.at(1, 2)[0], 6);
    EXPECT_EQ(fv.at(1, 3)[0], 7);
    EXPECT_EQ(fv.at(2, 0)[0], 8);
    EXPECT_EQ(fv.at(2, 2)[0], 10);
    EXPECT_EQ(fv.at(3, 0)[0], 12);
    EXPECT_EQ(fv.at(3, 3)[0], 15);
}

TEST(TestCanonicalLayout, Canonical3D)
{
    using field_t = covfie::field<covfie::backend::strided<
        covfie::vector::size3,
        covfie::backend::identity<covfie::vector::size1>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t{4u, 4u, 4u},
        field_t::backend_t::backend_t::configuration_t{}
    ));

    field_t::view_t fv(f);

    EXPECT_EQ(fv.at(0, 0, 0)[0], 0);
    EXPECT_EQ(fv.at(0, 3, 0)[0], 12);
    EXPECT_EQ(fv.at(1, 1, 0)[0], 20);
    EXPECT_EQ(fv.at(1, 2, 0)[0], 24);
    EXPECT_EQ(fv.at(1, 3, 0)[0], 28);
    EXPECT_EQ(fv.at(2, 0, 0)[0], 32);
    EXPECT_EQ(fv.at(2, 2, 0)[0], 40);
    EXPECT_EQ(fv.at(3, 0, 0)[0], 48);
    EXPECT_EQ(fv.at(3, 3, 0)[0], 60);
    EXPECT_EQ(fv.at(0, 0, 3)[0], 3);
    EXPECT_EQ(fv.at(0, 3, 3)[0], 15);
    EXPECT_EQ(fv.at(1, 1, 3)[0], 23);
    EXPECT_EQ(fv.at(1, 2, 3)[0], 27);
    EXPECT_EQ(fv.at(1, 3, 3)[0], 31);
    EXPECT_EQ(fv.at(2, 0, 3)[0], 35);
    EXPECT_EQ(fv.at(2, 2, 3)[0], 43);
    EXPECT_EQ(fv.at(3, 0, 3)[0], 51);
    EXPECT_EQ(fv.at(3, 3, 3)[0], 63);
}
