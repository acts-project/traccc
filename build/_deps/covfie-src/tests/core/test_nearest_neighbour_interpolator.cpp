/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cmath>
#include <cstddef>

#include <gtest/gtest.h>

#include <covfie/core/backend/primitive/identity.hpp>
#include <covfie/core/backend/transformer/nearest_neighbour.hpp>
#include <covfie/core/field.hpp>

TEST(TestNearestNeighbourInterpolator, Identity1Nto1F)
{
    using field_t = covfie::field<covfie::backend::nearest_neighbour<
        covfie::backend::identity<covfie::vector::int1>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({})
    ));
    field_t::view_t fv(f);

    EXPECT_EQ(fv.at(1.f)[0], 1);
    EXPECT_EQ(fv.at(5.f)[0], 5);
    EXPECT_EQ(fv.at(5.4f)[0], 5);
    EXPECT_EQ(fv.at(5.6f)[0], 6);
    EXPECT_EQ(fv.at(-1.f)[0], -1);
    EXPECT_EQ(fv.at(-1.49f)[0], -1);
    EXPECT_EQ(fv.at(-1.51f)[0], -2);
}

TEST(TestNearestNeighbourInterpolator, Identity2Nto2F)
{
    using field_t = covfie::field<covfie::backend::nearest_neighbour<
        covfie::backend::identity<covfie::vector::int2>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({})
    ));
    field_t::view_t fv(f);

    EXPECT_EQ(fv.at(-63.85f, 77.77f)[0], -64);
    EXPECT_EQ(fv.at(-63.85f, 77.77f)[1], 78);
    EXPECT_EQ(fv.at(0.66f, -94.72f)[0], 1);
    EXPECT_EQ(fv.at(0.66f, -94.72f)[1], -95);
    EXPECT_EQ(fv.at(-54.97f, -4.02f)[0], -55);
    EXPECT_EQ(fv.at(-54.97f, -4.02f)[1], -4);
    EXPECT_EQ(fv.at(28.02f, -12.28f)[0], 28);
    EXPECT_EQ(fv.at(28.02f, -12.28f)[1], -12);
    EXPECT_EQ(fv.at(54.06f, 63.58f)[0], 54);
    EXPECT_EQ(fv.at(54.06f, 63.58f)[1], 64);
    EXPECT_EQ(fv.at(-2.10f, 74.64f)[0], -2);
    EXPECT_EQ(fv.at(-2.10f, 74.64f)[1], 75);
    EXPECT_EQ(fv.at(-72.53f, -97.62f)[0], -73);
    EXPECT_EQ(fv.at(-72.53f, -97.62f)[1], -98);
    EXPECT_EQ(fv.at(-75.14f, -43.43f)[0], -75);
    EXPECT_EQ(fv.at(-75.14f, -43.43f)[1], -43);
    EXPECT_EQ(fv.at(89.90f, 98.13f)[0], 90);
    EXPECT_EQ(fv.at(89.90f, 98.13f)[1], 98);
    EXPECT_EQ(fv.at(-72.78f, -82.81f)[0], -73);
    EXPECT_EQ(fv.at(-72.78f, -82.81f)[1], -83);
    EXPECT_EQ(fv.at(80.43f, 89.60f)[0], 80);
    EXPECT_EQ(fv.at(80.43f, 89.60f)[1], 90);
    EXPECT_EQ(fv.at(34.30f, -46.85f)[0], 34);
    EXPECT_EQ(fv.at(34.30f, -46.85f)[1], -47);
    EXPECT_EQ(fv.at(40.52f, 53.92f)[0], 41);
    EXPECT_EQ(fv.at(40.52f, 53.92f)[1], 54);
    EXPECT_EQ(fv.at(28.47f, -24.85f)[0], 28);
    EXPECT_EQ(fv.at(28.47f, -24.85f)[1], -25);
    EXPECT_EQ(fv.at(26.75f, 55.20f)[0], 27);
    EXPECT_EQ(fv.at(26.75f, 55.20f)[1], 55);
    EXPECT_EQ(fv.at(-16.48f, 40.11f)[0], -16);
    EXPECT_EQ(fv.at(-16.48f, 40.11f)[1], 40);
    EXPECT_EQ(fv.at(37.80f, 41.26f)[0], 38);
    EXPECT_EQ(fv.at(37.80f, 41.26f)[1], 41);
    EXPECT_EQ(fv.at(-57.43f, 61.26f)[0], -57);
    EXPECT_EQ(fv.at(-57.43f, 61.26f)[1], 61);
    EXPECT_EQ(fv.at(36.65f, 66.34f)[0], 37);
    EXPECT_EQ(fv.at(36.65f, 66.34f)[1], 66);
    EXPECT_EQ(fv.at(-51.01f, 35.21f)[0], -51);
    EXPECT_EQ(fv.at(-51.01f, 35.21f)[1], 35);
}

TEST(TestNearestNeighbourInterpolator, Identity3Nto3F)
{
    using field_t = covfie::field<covfie::backend::nearest_neighbour<
        covfie::backend::identity<covfie::vector::int3>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({})
    ));
    field_t::view_t fv(f);

    EXPECT_EQ(fv.at(-40.52f, -45.92f, -66.71f)[0], -41);
    EXPECT_EQ(fv.at(-40.52f, -45.92f, -66.71f)[1], -46);
    EXPECT_EQ(fv.at(-40.52f, -45.92f, -66.71f)[2], -67);
    EXPECT_EQ(fv.at(-67.66f, 31.75f, 98.05f)[0], -68);
    EXPECT_EQ(fv.at(-67.66f, 31.75f, 98.05f)[1], 32);
    EXPECT_EQ(fv.at(-67.66f, 31.75f, 98.05f)[2], 98);
    EXPECT_EQ(fv.at(2.95f, -96.45f, -66.24f)[0], 3);
    EXPECT_EQ(fv.at(2.95f, -96.45f, -66.24f)[1], -96);
    EXPECT_EQ(fv.at(2.95f, -96.45f, -66.24f)[2], -66);
    EXPECT_EQ(fv.at(-99.85f, -90.42f, -45.33f)[0], -100);
    EXPECT_EQ(fv.at(-99.85f, -90.42f, -45.33f)[1], -90);
    EXPECT_EQ(fv.at(-99.85f, -90.42f, -45.33f)[2], -45);
    EXPECT_EQ(fv.at(-76.64f, -81.44f, -55.53f)[0], -77);
    EXPECT_EQ(fv.at(-76.64f, -81.44f, -55.53f)[1], -81);
    EXPECT_EQ(fv.at(-76.64f, -81.44f, -55.53f)[2], -56);
    EXPECT_EQ(fv.at(98.73f, -20.82f, -66.63f)[0], 99);
    EXPECT_EQ(fv.at(98.73f, -20.82f, -66.63f)[1], -21);
    EXPECT_EQ(fv.at(98.73f, -20.82f, -66.63f)[2], -67);
    EXPECT_EQ(fv.at(-27.35f, 44.53f, 86.10f)[0], -27);
    EXPECT_EQ(fv.at(-27.35f, 44.53f, 86.10f)[1], 45);
    EXPECT_EQ(fv.at(-27.35f, 44.53f, 86.10f)[2], 86);
    EXPECT_EQ(fv.at(64.53f, 50.67f, -89.84f)[0], 65);
    EXPECT_EQ(fv.at(64.53f, 50.67f, -89.84f)[1], 51);
    EXPECT_EQ(fv.at(64.53f, 50.67f, -89.84f)[2], -90);
    EXPECT_EQ(fv.at(-44.43f, -92.01f, -12.38f)[0], -44);
    EXPECT_EQ(fv.at(-44.43f, -92.01f, -12.38f)[1], -92);
    EXPECT_EQ(fv.at(-44.43f, -92.01f, -12.38f)[2], -12);
    EXPECT_EQ(fv.at(81.03f, -46.47f, 84.34f)[0], 81);
    EXPECT_EQ(fv.at(81.03f, -46.47f, 84.34f)[1], -46);
    EXPECT_EQ(fv.at(81.03f, -46.47f, 84.34f)[2], 84);
    EXPECT_EQ(fv.at(91.26f, -85.44f, -57.04f)[0], 91);
    EXPECT_EQ(fv.at(91.26f, -85.44f, -57.04f)[1], -85);
    EXPECT_EQ(fv.at(91.26f, -85.44f, -57.04f)[2], -57);
    EXPECT_EQ(fv.at(-74.90f, -27.67f, 87.74f)[0], -75);
    EXPECT_EQ(fv.at(-74.90f, -27.67f, 87.74f)[1], -28);
    EXPECT_EQ(fv.at(-74.90f, -27.67f, 87.74f)[2], 88);
    EXPECT_EQ(fv.at(72.70f, 56.80f, 70.44f)[0], 73);
    EXPECT_EQ(fv.at(72.70f, 56.80f, 70.44f)[1], 57);
    EXPECT_EQ(fv.at(72.70f, 56.80f, 70.44f)[2], 70);
    EXPECT_EQ(fv.at(-90.89f, 39.94f, 61.62f)[0], -91);
    EXPECT_EQ(fv.at(-90.89f, 39.94f, 61.62f)[1], 40);
    EXPECT_EQ(fv.at(-90.89f, 39.94f, 61.62f)[2], 62);
    EXPECT_EQ(fv.at(18.60f, -54.99f, 14.44f)[0], 19);
    EXPECT_EQ(fv.at(18.60f, -54.99f, 14.44f)[1], -55);
    EXPECT_EQ(fv.at(18.60f, -54.99f, 14.44f)[2], 14);
    EXPECT_EQ(fv.at(71.03f, -71.06f, 13.48f)[0], 71);
    EXPECT_EQ(fv.at(71.03f, -71.06f, 13.48f)[1], -71);
    EXPECT_EQ(fv.at(71.03f, -71.06f, 13.48f)[2], 13);
    EXPECT_EQ(fv.at(-9.30f, 88.57f, 80.03f)[0], -9);
    EXPECT_EQ(fv.at(-9.30f, 88.57f, 80.03f)[1], 89);
    EXPECT_EQ(fv.at(-9.30f, 88.57f, 80.03f)[2], 80);
    EXPECT_EQ(fv.at(78.96f, 97.57f, -12.28f)[0], 79);
    EXPECT_EQ(fv.at(78.96f, 97.57f, -12.28f)[1], 98);
    EXPECT_EQ(fv.at(78.96f, 97.57f, -12.28f)[2], -12);
    EXPECT_EQ(fv.at(24.47f, -56.02f, -44.02f)[0], 24);
    EXPECT_EQ(fv.at(24.47f, -56.02f, -44.02f)[1], -56);
    EXPECT_EQ(fv.at(24.47f, -56.02f, -44.02f)[2], -44);
    EXPECT_EQ(fv.at(-25.74f, -60.98f, -41.86f)[0], -26);
    EXPECT_EQ(fv.at(-25.74f, -60.98f, -41.86f)[1], -61);
    EXPECT_EQ(fv.at(-25.74f, -60.98f, -41.86f)[2], -42);
}
