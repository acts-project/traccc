/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <memory>

#include <cuda_runtime.h>

#include <covfie/core/vector.hpp>

namespace covfie::utility {
template <typename T>
struct to_cuda_channel_t {
};

template <>
struct to_cuda_channel_t<covfie::vector::float1> {
    using type = float;
};

template <>
struct to_cuda_channel_t<covfie::vector::float2> {
    using type = ::float2;
};

template <>
struct to_cuda_channel_t<covfie::vector::float3> {
    using type = ::float4;
};

template <>
struct to_cuda_channel_t<covfie::vector::float4> {
    using type = ::float4;
};
}
