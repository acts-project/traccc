/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <covfie/core/algebra/affine.hpp>
#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/primitive/constant.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/nearest_neighbour.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/field_view.hpp>

using backend_t1 = covfie::backend::affine<
    covfie::backend::nearest_neighbour<covfie::backend::strided<
        covfie::vector::size3,
        covfie::backend::array<covfie::vector::float3>>>>;

using backend_t2 = covfie::backend::nearest_neighbour<covfie::backend::strided<
    covfie::vector::size3,
    covfie::backend::array<covfie::vector::float3>>>;

using backend_t3 = covfie::backend::strided<
    covfie::vector::size3,
    covfie::backend::array<covfie::vector::float3>>;

using backend_t4 = covfie::backend::array<covfie::vector::float3>;

using approx_backend_t = covfie::backend::affine<
    covfie::backend::nearest_neighbour<covfie::backend::strided<
        covfie::vector::size3,
        covfie::backend::
            constant<covfie::vector::size1, covfie::vector::float3>>>>;

template covfie::field_view<approx_backend_t>::output_t
    covfie::field_view<approx_backend_t>::at<
        approx_backend_t::contravariant_input_t::scalar_t,
        approx_backend_t::contravariant_input_t::scalar_t,
        approx_backend_t::contravariant_input_t::scalar_t>(
        approx_backend_t::contravariant_input_t::scalar_t,
        approx_backend_t::contravariant_input_t::scalar_t,
        approx_backend_t::contravariant_input_t::scalar_t
    ) const;
template covfie::field_view<backend_t1>::output_t
    covfie::field_view<backend_t1>::at<
        backend_t1::contravariant_input_t::scalar_t,
        backend_t1::contravariant_input_t::scalar_t,
        backend_t1::contravariant_input_t::scalar_t>(
        backend_t1::contravariant_input_t::scalar_t,
        backend_t1::contravariant_input_t::scalar_t,
        backend_t1::contravariant_input_t::scalar_t
    ) const;
template covfie::field_view<backend_t2>::output_t
    covfie::field_view<backend_t2>::at<
        backend_t2::contravariant_input_t::scalar_t,
        backend_t2::contravariant_input_t::scalar_t,
        backend_t2::contravariant_input_t::scalar_t>(
        backend_t2::contravariant_input_t::scalar_t,
        backend_t2::contravariant_input_t::scalar_t,
        backend_t2::contravariant_input_t::scalar_t
    ) const;
template covfie::field_view<backend_t3>::output_t
    covfie::field_view<backend_t3>::at<
        backend_t3::contravariant_input_t::scalar_t>(
        backend_t3::contravariant_input_t::scalar_t,
        backend_t3::contravariant_input_t::scalar_t,
        backend_t3::contravariant_input_t::scalar_t
    ) const;
template covfie::field_view<backend_t4>::output_t
    covfie::field_view<backend_t4>::at<
        backend_t4::contravariant_input_t::scalar_t>(
        backend_t4::contravariant_input_t::scalar_t
    ) const;
