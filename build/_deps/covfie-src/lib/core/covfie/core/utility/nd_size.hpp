/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <covfie/core/array.hpp>

namespace covfie::utility {
template <std::size_t N>
using nd_size = array::array<std::size_t, N>;
}
