/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <sstream>

#include <cuda_runtime_api.h>

#define cudaErrorCheck(r)                                                      \
    {                                                                          \
        _cudaErrorCheck((r), __FILE__, __LINE__);                              \
    }

inline void _cudaErrorCheck(cudaError_t code, const char * file, int line)
{
    if (code != cudaSuccess) {
        std::stringstream ss;

        ss << "[" << file << ":" << line
           << "] CUDA error: " << cudaGetErrorString(code);

        throw std::runtime_error(ss.str());
    }
}
