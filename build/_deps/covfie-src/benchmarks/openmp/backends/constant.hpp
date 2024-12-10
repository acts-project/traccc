/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <covfie/core/backend/primitive/constant.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/vector.hpp>

template <typename S, std::size_t N, std::size_t M>
struct Constant {
    using backend_t = covfie::backend::constant<
        covfie::vector::vector_d<S, N>,
        covfie::vector::vector_d<S, M>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(typename backend_t::configuration_t{0.f}
        );
    }
};
