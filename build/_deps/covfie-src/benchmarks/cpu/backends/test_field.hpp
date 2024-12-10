/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <covfie/benchmark/test_field.hpp>
#include <covfie/core/backend/primitive/constant.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/morton.hpp>
#include <covfie/core/backend/transformer/nearest_neighbour.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/vector.hpp>

struct FieldConstant {
    using backend_t = covfie::backend::constant<
        covfie::vector::vector_d<float, 3>,
        covfie::vector::vector_d<float, 3>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(covfie::make_parameter_pack(
            backend_t::configuration_t{0.f, 0.f, 2.f}
        ));
    }
};

struct InterpolateNN {
    template <typename T>
    using apply = covfie::backend::nearest_neighbour<T>;
};

struct InterpolateLin {
    template <typename T>
    using apply = covfie::backend::linear<T>;
};

struct LayoutStride {
    template <typename T>
    using apply = covfie::backend::strided<covfie::vector::size3, T>;
};

struct LayoutMortonBMI2 {
    template <typename T>
    using apply = covfie::backend::morton<covfie::vector::size3, T, true>;
};

struct LayoutMortonNaive {
    template <typename T>
    using apply = covfie::backend::morton<covfie::vector::size3, T, false>;
};

template <typename Interpolator, typename Layout>
struct Field {
    using backend_t = covfie::backend::affine<
        typename Interpolator::template apply<typename Layout::template apply<

            covfie::backend::array<covfie::vector::float3>>>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(get_test_field());
    }
};

struct FieldIntBase {
    using backend_t = covfie::backend::strided<
        covfie::vector::size3,
        covfie::backend::array<covfie::vector::float3>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(covfie::make_parameter_pack(
            get_test_field().backend().get_backend().get_backend()
        ));
    }
};

struct FieldIntMorton {
    using backend_t = covfie::backend::morton<
        covfie::vector::size3,
        covfie::backend::array<covfie::vector::float3>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(covfie::make_parameter_pack(
            get_test_field().backend().get_backend().get_backend()
        ));
    }
};
