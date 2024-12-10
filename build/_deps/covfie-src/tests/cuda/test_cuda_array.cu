/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <optional>

#include <gtest/gtest.h>

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/field_view.hpp>
#include <covfie/core/utility/nd_map.hpp>
#include <covfie/core/vector.hpp>
#include <covfie/cuda/backend/primitive/cuda_device_array.hpp>

#include "retrieve_vector.hpp"

template <std::size_t N, typename B>
class TestLookupGeneric : public ::testing::Test
{
protected:
    void SetUp() override
    {
        using canonical_backend_t = covfie::backend::strided<
            covfie::vector::size3,
            covfie::backend::array<covfie::vector::size3>>;

        covfie::array::array<std::size_t, 3> sizes{10UL, 10UL, 10UL};

        covfie::field<canonical_backend_t> f(covfie::make_parameter_pack(
            canonical_backend_t::configuration_t{sizes}
        ));
        covfie::field_view<canonical_backend_t> fv(f);

        covfie::utility::nd_map<decltype(sizes)>(
            [&fv](decltype(sizes) t) { fv.at(t) = t; }, sizes
        );

        m_field = covfie::field<B>(f);
    }

    std::optional<covfie::field<B>> m_field;
};

template <typename B>
using TestCudaLookupIntegerIndexed1D = TestLookupGeneric<1, B>;

using BackendsInteger1D = ::testing::Types<covfie::backend::strided<
    covfie::vector::size1,
    covfie::backend::cuda_device_array<covfie::vector::size1>>>;

TYPED_TEST_SUITE(TestCudaLookupIntegerIndexed1D, BackendsInteger1D);

template <typename B>
using TestCudaLookupIntegerIndexed2D = TestLookupGeneric<2, B>;

using BackendsInteger2D = ::testing::Types<covfie::backend::strided<
    covfie::vector::size2,
    covfie::backend::cuda_device_array<covfie::vector::size2>>>;

TYPED_TEST_SUITE(TestCudaLookupIntegerIndexed2D, BackendsInteger2D);

template <typename B>
using TestCudaLookupIntegerIndexed3D = TestLookupGeneric<3, B>;

using BackendsInteger3D = ::testing::Types<covfie::backend::strided<
    covfie::vector::size3,
    covfie::backend::cuda_device_array<covfie::vector::size3>>>;

TYPED_TEST_SUITE(TestCudaLookupIntegerIndexed3D, BackendsInteger3D);

TYPED_TEST(TestCudaLookupIntegerIndexed3D, LookUp)
{
    for (std::size_t x = 0; x < 10; ++x) {
        for (std::size_t y = 0; y < 10; ++y) {
            for (std::size_t z = 0; z < 10; ++z) {
                std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                    retrieve_vector<covfie::field<TypeParam>>(
                        *(this->m_field), {x, y, z}
                    );

                EXPECT_EQ(o[0], x);
                EXPECT_EQ(o[1], y);
                EXPECT_EQ(o[2], z);
            }
        }
    }
}
