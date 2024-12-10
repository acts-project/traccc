/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstddef>

#include <covfie/core/array.hpp>

namespace covfie::vector {
template <typename _type, std::size_t _size>
struct vector_d {
    using type = _type;
    static constexpr std::size_t size = _size;
};

template <typename _vector_d>
struct array_vector_d {
    using vector_d = _vector_d;
    using scalar_t = typename vector_d::type;
    static constexpr std::size_t dimensions = vector_d::size;
    using vector_t = array::array<typename vector_d::type, vector_d::size>;
};

template <typename _vector_d>
struct array_reference_vector_d {
    using vector_d = _vector_d;
    using scalar_t = typename vector_d::type;
    static constexpr std::size_t dimensions = vector_d::size;
    using vector_t = std::add_lvalue_reference_t<
        array::array<typename vector_d::type, vector_d::size>>;
};

template <typename _vector_d>
struct scalar_d {
    static_assert(
        _vector_d::size == 1,
        "Scalar type is only usable with vectors of size 1."
    );

    using vector_d = _vector_d;
    using scalar_t = typename vector_d::type;
    static constexpr std::size_t dimensions = vector_d::size;
    using vector_t = scalar_t;
};

using float1 = vector_d<float, 1>;
using float2 = vector_d<float, 2>;
using float3 = vector_d<float, 3>;
using float4 = vector_d<float, 4>;

using double1 = vector_d<double, 1>;
using double2 = vector_d<double, 2>;
using double3 = vector_d<double, 3>;
using double4 = vector_d<double, 4>;

using int1 = vector_d<int, 1>;
using int2 = vector_d<int, 2>;
using int3 = vector_d<int, 3>;
using int4 = vector_d<int, 4>;

using uint1 = vector_d<unsigned int, 1>;
using uint2 = vector_d<unsigned int, 2>;
using uint3 = vector_d<unsigned int, 3>;
using uint4 = vector_d<unsigned int, 4>;

using long1 = vector_d<long, 1>;
using long2 = vector_d<long, 2>;
using long3 = vector_d<long, 3>;
using long4 = vector_d<long, 4>;

using ulong1 = vector_d<unsigned long, 1>;
using ulong2 = vector_d<unsigned long, 2>;
using ulong3 = vector_d<unsigned long, 3>;
using ulong4 = vector_d<unsigned long, 4>;

using size1 = vector_d<std::size_t, 1>;
using size2 = vector_d<std::size_t, 2>;
using size3 = vector_d<std::size_t, 3>;
using size4 = vector_d<std::size_t, 4>;
}
