/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <covfie/cuda/error_check.hpp>

template <typename F>
__global__ void retrieve_vector_kernel(
    typename F::view_t v,
    typename F::coordinate_t c,
    std::decay_t<typename F::output_t> * o
)
{
    *o = v.at(c);
}

template <typename F>
std::decay_t<typename F::output_t>
retrieve_vector(typename F::view_t v, typename F::coordinate_t c)
{
    std::decay_t<typename F::output_t> * rv_d;
    std::decay_t<typename F::output_t> rv_h;

    cudaErrorCheck(cudaMalloc(&rv_d, sizeof(std::decay_t<typename F::output_t>))
    );

    retrieve_vector_kernel<F><<<1, 1>>>(v, c, rv_d);

    cudaErrorCheck(cudaGetLastError());
    cudaErrorCheck(cudaDeviceSynchronize());

    cudaErrorCheck(cudaMemcpy(
        &rv_h,
        rv_d,
        sizeof(std::decay_t<typename F::output_t>),
        cudaMemcpyDeviceToHost
    ));
    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaFree(rv_d));

    return rv_h;
}
