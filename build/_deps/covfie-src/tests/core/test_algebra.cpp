/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <gtest/gtest.h>

#include <covfie/core/algebra/affine.hpp>
#include <covfie/core/algebra/matrix.hpp>
#include <covfie/core/algebra/vector.hpp>

TEST(TestAlgebra, VectorArrayInit1F)
{
    covfie::algebra::vector<1, float> v(covfie::array::array<float, 1>{17.2f});

    EXPECT_FLOAT_EQ(v(0), 17.2f);
}

TEST(TestAlgebra, VectorVariadicInit1F)
{
    covfie::algebra::vector<1, float> v(49.2f);

    EXPECT_FLOAT_EQ(v(0), 49.2f);
}

TEST(TestAlgebra, VectorArrayInit1D)
{
    covfie::algebra::vector<1, double> v(covfie::array::array<double, 1>{21.5});

    EXPECT_DOUBLE_EQ(v(0), 21.5);
}

TEST(TestAlgebra, VectorVariadicInit1D)
{
    covfie::algebra::vector<1, double> v(84.2);

    EXPECT_DOUBLE_EQ(v(0), 84.2);
}

TEST(TestAlgebra, VectorArrayInit2F)
{
    covfie::algebra::vector<2, float> v(covfie::array::array<float, 2>{
        17.2f, 92.4f});

    EXPECT_FLOAT_EQ(v(0), 17.2f);
    EXPECT_FLOAT_EQ(v(1), 92.4f);
}

TEST(TestAlgebra, VectorVariadicInit2F)
{
    covfie::algebra::vector<2, float> v(49.2f, 40.2f);

    EXPECT_FLOAT_EQ(v(0), 49.2f);
    EXPECT_FLOAT_EQ(v(1), 40.2f);
}

TEST(TestAlgebra, VectorArrayInit2D)
{
    covfie::algebra::vector<2, double> v(covfie::array::array<double, 2>{
        21.5, 11.8});

    EXPECT_DOUBLE_EQ(v(0), 21.5);
    EXPECT_DOUBLE_EQ(v(1), 11.8);
}

TEST(TestAlgebra, VectorVariadicInit2D)
{
    covfie::algebra::vector<2, double> v(84.2, 77.3);

    EXPECT_DOUBLE_EQ(v(0), 84.2);
    EXPECT_DOUBLE_EQ(v(1), 77.3);
}

TEST(TestAlgebra, VectorArrayInit3F)
{
    covfie::algebra::vector<3, float> v(covfie::array::array<float, 3>{
        17.2f, 92.4f, 39.5f});

    EXPECT_FLOAT_EQ(v(0), 17.2f);
    EXPECT_FLOAT_EQ(v(1), 92.4f);
    EXPECT_FLOAT_EQ(v(2), 39.5f);
}

TEST(TestAlgebra, VectorVariadicInit3F)
{
    covfie::algebra::vector<3, float> v(49.2f, 40.2f, 80.2f);

    EXPECT_FLOAT_EQ(v(0), 49.2f);
    EXPECT_FLOAT_EQ(v(1), 40.2f);
    EXPECT_FLOAT_EQ(v(2), 80.2f);
}

TEST(TestAlgebra, VectorArrayInit3D)
{
    covfie::algebra::vector<3, double> v(covfie::array::array<double, 3>{
        21.5, 11.8, 28.2});

    EXPECT_DOUBLE_EQ(v(0), 21.5);
    EXPECT_DOUBLE_EQ(v(1), 11.8);
    EXPECT_DOUBLE_EQ(v(2), 28.2);
}

TEST(TestAlgebra, VectorVariadicInit3D)
{
    covfie::algebra::vector<3, double> v(84.2, 77.3, 66.1);

    EXPECT_DOUBLE_EQ(v(0), 84.2);
    EXPECT_DOUBLE_EQ(v(1), 77.3);
    EXPECT_DOUBLE_EQ(v(2), 66.1);
}

TEST(TestAlgebra, VectorAssignment1F)
{
    covfie::algebra::vector<1, float> v;

    v(0) = 5.2f;

    EXPECT_FLOAT_EQ(v(0), 5.2f);
}

TEST(TestAlgebra, VectorAssignment1D)
{
    covfie::algebra::vector<1, double> v;

    v(0) = 5.3;

    EXPECT_DOUBLE_EQ(v(0), 5.3);
}

TEST(TestAlgebra, VectorAssignment2F)
{
    covfie::algebra::vector<2, float> v;

    v(0) = 5.2f;
    v(1) = 5.4f;

    EXPECT_FLOAT_EQ(v(0), 5.2f);
    EXPECT_FLOAT_EQ(v(1), 5.4f);
}

TEST(TestAlgebra, VectorAssignment2D)
{
    covfie::algebra::vector<2, double> v;

    v(0) = 5.3;
    v(1) = 5.5;

    EXPECT_DOUBLE_EQ(v(0), 5.3);
    EXPECT_DOUBLE_EQ(v(1), 5.5);
}

TEST(TestAlgebra, VectorAssignment3F)
{
    covfie::algebra::vector<3, float> v;

    v(0) = 5.2f;
    v(1) = 5.6f;
    v(2) = 5.8f;

    EXPECT_FLOAT_EQ(v(0), 5.2f);
    EXPECT_FLOAT_EQ(v(1), 5.6f);
    EXPECT_FLOAT_EQ(v(2), 5.8f);
}

TEST(TestAlgebra, VectorAssignment3D)
{
    covfie::algebra::vector<3, double> v;

    v(0) = 5.3;
    v(1) = 6.3;
    v(2) = 7.3;

    EXPECT_DOUBLE_EQ(v(0), 5.3);
    EXPECT_DOUBLE_EQ(v(1), 6.3);
    EXPECT_DOUBLE_EQ(v(2), 7.3);
}

TEST(TestAlgebra, MatrixInit1x1F)
{
    covfie::algebra::matrix<1, 1, float> m(
        covfie::array::array<covfie::array::array<float, 1>, 1>{{{5.2f}}}
    );

    EXPECT_FLOAT_EQ(m(0, 0), 5.2f);
}

TEST(TestAlgebra, MatrixInit2x2F)
{
    covfie::algebra::matrix<2, 2, float> m(
        covfie::array::array<covfie::array::array<float, 2>, 2>{
            {{0.5f, 10.5f}, {1.5f, 11.5f}}}
    );

    EXPECT_FLOAT_EQ(m(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(m(0, 1), 10.5f);
    EXPECT_FLOAT_EQ(m(1, 0), 1.5f);
    EXPECT_FLOAT_EQ(m(1, 1), 11.5f);
}

TEST(TestAlgebra, MatrixInit3x3F)
{
    covfie::algebra::matrix<3, 3, float> m(
        covfie::array::array<covfie::array::array<float, 3>, 3>{
            {{0.5f, 10.5f, 20.5f}, {1.5f, 11.5f, 21.5f}, {2.5f, 12.5f, 22.5f}}}
    );

    EXPECT_FLOAT_EQ(m(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(m(0, 1), 10.5f);
    EXPECT_FLOAT_EQ(m(0, 2), 20.5f);
    EXPECT_FLOAT_EQ(m(1, 0), 1.5f);
    EXPECT_FLOAT_EQ(m(1, 1), 11.5f);
    EXPECT_FLOAT_EQ(m(1, 2), 21.5f);
    EXPECT_FLOAT_EQ(m(2, 0), 2.5f);
    EXPECT_FLOAT_EQ(m(2, 1), 12.5f);
    EXPECT_FLOAT_EQ(m(2, 2), 22.5f);
}

TEST(TestAlgebra, MatrixInit3x2F)
{
    covfie::algebra::matrix<3, 2, float> m(
        covfie::array::array<covfie::array::array<float, 2>, 3>{
            {{0.5f, 10.5f}, {1.5f, 11.5f}, {2.5f, 12.5f}}}
    );

    EXPECT_FLOAT_EQ(m(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(m(0, 1), 10.5f);
    EXPECT_FLOAT_EQ(m(1, 0), 1.5f);
    EXPECT_FLOAT_EQ(m(1, 1), 11.5f);
    EXPECT_FLOAT_EQ(m(2, 0), 2.5f);
    EXPECT_FLOAT_EQ(m(2, 1), 12.5f);
}

TEST(TestAlgebra, AffineInit1F)
{
    covfie::algebra::affine<1, float> m(
        covfie::array::array<covfie::array::array<float, 2>, 1>{{{0.5f, 10.5f}}}
    );

    EXPECT_FLOAT_EQ(m(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(m(0, 1), 10.5f);
}

TEST(TestAlgebra, AffineInit2F)
{
    covfie::algebra::affine<2, float> m(
        covfie::array::array<covfie::array::array<float, 3>, 2>{
            {{0.5f, 10.5f, 20.5f}, {1.5f, 11.5f, 21.5f}}}
    );

    EXPECT_FLOAT_EQ(m(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(m(0, 1), 10.5f);
    EXPECT_FLOAT_EQ(m(0, 2), 20.5f);
    EXPECT_FLOAT_EQ(m(1, 0), 1.5f);
    EXPECT_FLOAT_EQ(m(1, 1), 11.5f);
    EXPECT_FLOAT_EQ(m(1, 2), 21.5f);
}

TEST(TestAlgebra, AffineInit3F)
{
    covfie::algebra::affine<3, float> m(
        covfie::array::array<covfie::array::array<float, 4>, 3>{
            {{0.5f, 10.5f, 20.5f, 30.5f},
             {1.5f, 11.5f, 21.5f, 31.5f},
             {2.5f, 12.5f, 22.5f, 32.5f}}}
    );

    EXPECT_FLOAT_EQ(m(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(m(0, 1), 10.5f);
    EXPECT_FLOAT_EQ(m(0, 2), 20.5f);
    EXPECT_FLOAT_EQ(m(0, 3), 30.5f);
    EXPECT_FLOAT_EQ(m(1, 0), 1.5f);
    EXPECT_FLOAT_EQ(m(1, 1), 11.5f);
    EXPECT_FLOAT_EQ(m(1, 2), 21.5f);
    EXPECT_FLOAT_EQ(m(1, 3), 31.5f);
    EXPECT_FLOAT_EQ(m(2, 0), 2.5f);
    EXPECT_FLOAT_EQ(m(2, 1), 12.5f);
    EXPECT_FLOAT_EQ(m(2, 2), 22.5f);
    EXPECT_FLOAT_EQ(m(2, 3), 32.5f);
}

TEST(TestAlgebra, MatrixMatrixMultiplication1x1x1F)
{
    covfie::algebra::matrix<1, 1, float> m1(
        covfie::array::array<covfie::array::array<float, 1>, 1>{{{0.5f}}}
    );
    covfie::algebra::matrix<1, 1, float> m2(
        covfie::array::array<covfie::array::array<float, 1>, 1>{{{1.5f}}}
    );

    covfie::algebra::matrix<1, 1, float> mr1 = m1 * m2;
    covfie::algebra::matrix<1, 1, float> mr2 = m2 * m1;

    EXPECT_FLOAT_EQ(mr1(0, 0), 0.75f);
    EXPECT_FLOAT_EQ(mr2(0, 0), 0.75f);
}

TEST(TestAlgebra, MatrixMatrixMultiplication2x2x2F)
{
    covfie::algebra::matrix<2, 2, float> m1(
        covfie::array::array<covfie::array::array<float, 2>, 2>{
            {{0.5f, 10.5f}, {1.5f, 11.5f}}}
    );
    covfie::algebra::matrix<2, 2, float> m2(
        covfie::array::array<covfie::array::array<float, 2>, 2>{
            {{10.5f, 110.5f}, {11.5f, 111.5f}}}
    );

    covfie::algebra::matrix<2, 2, float> mr1 = m1 * m2;
    covfie::algebra::matrix<2, 2, float> mr2 = m2 * m1;

    EXPECT_FLOAT_EQ(mr1(0, 0), 126.f);
    EXPECT_FLOAT_EQ(mr1(0, 1), 1226.f);
    EXPECT_FLOAT_EQ(mr1(1, 0), 148.f);
    EXPECT_FLOAT_EQ(mr1(1, 1), 1448.f);

    EXPECT_FLOAT_EQ(mr2(0, 0), 171.f);
    EXPECT_FLOAT_EQ(mr2(0, 1), 1381.f);
    EXPECT_FLOAT_EQ(mr2(1, 0), 173.f);
    EXPECT_FLOAT_EQ(mr2(1, 1), 1403.f);
}

TEST(TestAlgebra, MatrixMatrixMultiplication3x3x3F)
{
    covfie::algebra::matrix<3, 3, float> m1(
        covfie::array::array<covfie::array::array<float, 3>, 3>{
            {{0.5f, 10.5f, 20.5f}, {1.5f, 11.5f, 21.5f}, {2.5f, 12.5f, 22.5f}}}
    );
    covfie::algebra::matrix<3, 3, float> m2(
        covfie::array::array<covfie::array::array<float, 3>, 3>{
            {{10.5f, 110.5f, 120.5f},
             {11.5f, 111.5f, 121.5f},
             {12.5f, 112.5f, 122.5f}}}
    );

    covfie::algebra::matrix<3, 3, float> mr1 = m1 * m2;
    covfie::algebra::matrix<3, 3, float> mr2 = m2 * m1;

    EXPECT_FLOAT_EQ(mr1(0, 0), 382.25f);
    EXPECT_FLOAT_EQ(mr1(0, 1), 3532.25f);
    EXPECT_FLOAT_EQ(mr1(0, 2), 3847.25f);
    EXPECT_FLOAT_EQ(mr1(1, 0), 416.75f);
    EXPECT_FLOAT_EQ(mr1(1, 1), 3866.75f);
    EXPECT_FLOAT_EQ(mr1(1, 2), 4211.75f);
    EXPECT_FLOAT_EQ(mr1(2, 0), 451.25f);
    EXPECT_FLOAT_EQ(mr1(2, 1), 4201.25f);
    EXPECT_FLOAT_EQ(mr1(2, 2), 4576.25f);

    EXPECT_FLOAT_EQ(mr2(0, 0), 472.25f);
    EXPECT_FLOAT_EQ(mr2(0, 1), 2887.25f);
    EXPECT_FLOAT_EQ(mr2(0, 2), 5302.25f);
    EXPECT_FLOAT_EQ(mr2(1, 0), 476.75f);
    EXPECT_FLOAT_EQ(mr2(1, 1), 2921.75f);
    EXPECT_FLOAT_EQ(mr2(1, 2), 5366.75f);
    EXPECT_FLOAT_EQ(mr2(2, 0), 481.25f);
    EXPECT_FLOAT_EQ(mr2(2, 1), 2956.25f);
    EXPECT_FLOAT_EQ(mr2(2, 2), 5431.25f);
}

TEST(TestAlgebra, MatrixMatrixMultiplication3x2x4F)
{
    covfie::algebra::matrix<3, 2, float> m1(
        covfie::array::array<covfie::array::array<float, 2>, 3>{
            {{0.5f, 10.5f}, {1.5f, 11.5f}, {2.5f, 12.5f}}}
    );
    covfie::algebra::matrix<2, 4, float> m2(
        covfie::array::array<covfie::array::array<float, 4>, 2>{
            {{10.5f, 110.5f, 120.5f, 130.5f}, {11.5f, 111.5f, 121.5f, 131.5f}}}
    );

    covfie::algebra::matrix<3, 4, float> mr = m1 * m2;

    EXPECT_FLOAT_EQ(mr(0, 0), 126.f);
    EXPECT_FLOAT_EQ(mr(0, 1), 1226.f);
    EXPECT_FLOAT_EQ(mr(0, 2), 1336.f);
    EXPECT_FLOAT_EQ(mr(0, 3), 1446.f);
    EXPECT_FLOAT_EQ(mr(1, 0), 148.f);
    EXPECT_FLOAT_EQ(mr(1, 1), 1448.f);
    EXPECT_FLOAT_EQ(mr(1, 2), 1578.f);
    EXPECT_FLOAT_EQ(mr(1, 3), 1708.f);
    EXPECT_FLOAT_EQ(mr(2, 0), 170.f);
    EXPECT_FLOAT_EQ(mr(2, 1), 1670.f);
    EXPECT_FLOAT_EQ(mr(2, 2), 1820.f);
    EXPECT_FLOAT_EQ(mr(2, 3), 1970.f);
}

TEST(TestAlgebra, MatrixVectorMultiplication3x3x1F)
{
    covfie::algebra::matrix<3, 3, float> m(
        covfie::array::array<covfie::array::array<float, 3>, 3>{
            {{0.5f, 10.5f, 20.5f}, {1.5f, 11.5f, 21.5f}, {2.5f, 12.5f, 22.5f}}}
    );
    covfie::algebra::vector<3, float> v(4.2f, 9.1f, 5.5f);

    covfie::algebra::vector<3, float> r = m * v;

    EXPECT_FLOAT_EQ(r(0), 210.4f);
    EXPECT_FLOAT_EQ(r(1), 229.2f);
    EXPECT_FLOAT_EQ(r(2), 248.0f);
}

TEST(TestAlgebra, AffineVectorMultiplication1F)
{
    covfie::algebra::affine<1, float> m(
        covfie::array::array<covfie::array::array<float, 2>, 1>{{{0.5f, 10.5f}}}
    );
    covfie::algebra::vector<1, float> v(4.2f);

    covfie::algebra::vector<1, float> r = m * v;

    EXPECT_FLOAT_EQ(r(0), 12.6f);
}

TEST(TestAlgebra, AffineVectorMultiplication2F)
{
    covfie::algebra::affine<2, float> m(
        covfie::array::array<covfie::array::array<float, 3>, 2>{
            {{0.5f, 10.5f, 20.5f}, {1.5f, 11.5f, 21.5f}}}
    );
    covfie::algebra::vector<2, float> v(4.2f, 7.1f);

    covfie::algebra::vector<2, float> r = m * v;

    EXPECT_FLOAT_EQ(r(0), 97.15f);
    EXPECT_FLOAT_EQ(r(1), 109.45f);
}

TEST(TestAlgebra, AffineVectorMultiplication3F)
{
    covfie::algebra::affine<3, float> m(
        covfie::array::array<covfie::array::array<float, 4>, 3>{
            {{0.5f, 10.5f, 20.5f, 30.5f},
             {1.5f, 11.5f, 21.5f, 31.5f},
             {2.5f, 12.5f, 22.5f, 32.5f}}}
    );
    covfie::algebra::vector<3, float> v(4.2f, 7.1f, 5.9f);

    covfie::algebra::vector<3, float> r = m * v;

    EXPECT_FLOAT_EQ(r(0), 228.1f);
    EXPECT_FLOAT_EQ(r(1), 246.3f);
    EXPECT_FLOAT_EQ(r(2), 264.5f);
}
