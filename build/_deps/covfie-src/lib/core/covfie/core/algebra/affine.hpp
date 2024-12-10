/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstddef>

#include <covfie/core/algebra/matrix.hpp>
#include <covfie/core/algebra/vector.hpp>
#include <covfie/core/array.hpp>
#include <covfie/core/qualifiers.hpp>

namespace covfie::algebra {
template <std::size_t N, typename T = float, typename I = std::size_t>
struct affine : public matrix<N, N + 1, T, I> {
    affine() = default;
    affine(affine &&) = default;
    affine(const affine &) = default;
    affine & operator=(const affine &) = default;
    affine & operator=(affine &&) = default;

    COVFIE_DEVICE affine(const matrix<N, N + 1, T, I> & o)
        : matrix<N, N + 1, T, I>(o)
    {
    }

    COVFIE_DEVICE vector<N, T, I> operator*(const vector<N, T, I> & v) const
    {
        vector<N + 1, T, I> r;

        for (I i = 0; i < N; ++i) {
            r(i) = v(i);
        }

        r(N) = static_cast<T>(1.);

        return matrix<N, N + 1, T, I>::operator*(r);
    }

    COVFIE_DEVICE affine<N, T, I> operator*(const affine<N, T, I> & m) const
    {
        matrix<N + 1, N + 1, T, I> m1, m2;

        for (I i = 0; i < N; ++i) {
            for (I j = 0; j < N + 1; ++j) {
                m1(i, j) = this->operator()(i, j);
                m2(i, j) = m.operator()(i, j);
            }
        }

        for (I j = 0; j < N + 1; ++j) {
            if (j == N) {
                m1(N, j) = 1.f;
                m2(N, j) = 1.f;
            } else {
                m1(N, j) = 0.f;
                m2(N, j) = 0.f;
            }
        }

        matrix<N + 1, N + 1, T, I> r = m1 * m2;
        matrix<N, N + 1, T, I> o;

        for (I i = 0; i < N; ++i) {
            for (I j = 0; j < N + 1; ++j) {
                o(i, j) = r(i, j);
            }
        }

        return o;
    }

    template <typename... Args>
    COVFIE_DEVICE static affine<N, T, I> translation(const Args &... args)
    {
        static_assert(
            (std::is_convertible_v<Args, T> && ...),
            "Translation arguments must be convertible to transformation "
            "matrix elements."
        );
        static_assert(
            sizeof...(Args) == N,
            "Translation must have exactly as many arguments as the dimensions "
            "of the matrix."
        );

        array::array<T, N> arr{args...};

        matrix<N, N + 1, T, I> result = matrix<N, N + 1, T, I>::identity();

        for (I i = 0; i < N; ++i) {
            result(i, N) = arr[i];
        }

        return result;
    }

    template <typename... Args>
    COVFIE_DEVICE static affine<N, T, I> scaling(const Args &... args)
    {
        static_assert(
            (std::is_convertible_v<Args, T> && ...),
            "Scaling arguments must be convertible to transformation matrix "
            "elements."
        );
        static_assert(
            sizeof...(Args) == N,
            "Scaling must have exactly as many arguments as the dimensions of "
            "the matrix."
        );

        array::array<T, N> arr{args...};

        matrix<N, N + 1, T, I> result = matrix<N, N + 1, T, I>::identity();

        for (I i = 0; i < N; ++i) {
            result(i, i) = arr[i];
        }

        return result;
    }
};
}
