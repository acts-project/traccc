/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <memory>

#include <covfie/core/backend/primitive/array.hpp>

namespace covfie::backend {
template <
    concepts::vector_descriptor _output_vector_t,
    typename _index_t = std::size_t>
using c_array = array<_output_vector_t, _index_t>;
}
