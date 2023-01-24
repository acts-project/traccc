/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cuda_runtime.h>

#include "traccc/cuda/utils/definitions.hpp"

#ifdef __CUDACC__
__global__ void set_memory_kernel(int *v, std::size_t s) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < s) {
        v[tid] = tid;
    }
}

__global__ void get_memory_kernel(int *v, int *o, std::size_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        *o = v[n];
    }
}
#endif

class alg1 {
    public:
    using result_type = std::tuple<int *, std::size_t>;
    using config_type = std::size_t;
    using argument_type = std::monostate;

    static std::tuple<cudaGraph_t, cudaGraphNode_t, result_type> create_graph(
        config_type c, argument_type) {
        cudaGraph_t g;

        CUDA_ERROR_CHECK(cudaGraphCreate(&g, 0));

        cudaGraphNode_t allocation_node;

        cudaMemAllocNodeParams alloc_params;
        memset(&alloc_params, 0, sizeof(alloc_params));
        alloc_params.bytesize = c * sizeof(int);
        alloc_params.poolProps.allocType = cudaMemAllocationTypePinned;
        alloc_params.poolProps.location.id = 0;
        alloc_params.poolProps.location.type = cudaMemLocationTypeDevice;

        CUDA_ERROR_CHECK(cudaGraphAddMemAllocNode(&allocation_node, g, nullptr,
                                                  0, &alloc_params));

        return {g, allocation_node,
                result_type{reinterpret_cast<int *>(alloc_params.dptr), c}};
    }
};

class alg2 {
    public:
    using result_type = std::tuple<int *, std::size_t>;
    using config_type = std::monostate;
    using argument_type = std::tuple<int *, std::size_t>;

    static std::tuple<cudaGraphNode_t, result_type> append_graph(
        cudaGraph_t g, cudaGraphNode_t n, config_type, argument_type a) {
        cudaGraphNode_t kernel_node;

        cudaGraphNode_t kernel_node_deps[1] = {n};

        void *kernel_launch_params[2] = {&std::get<0>(a), &std::get<1>(a)};

        cudaKernelNodeParams kernel_params;

#ifdef __CUDACC__
        kernel_params.func = reinterpret_cast<void *>(set_memory_kernel);
#else
        kernel_params.func = nullptr;
#endif

        kernel_params.gridDim = {
            static_cast<unsigned int>(std::get<1>(a) / 512 + 1)};
        kernel_params.blockDim = {512u};
        kernel_params.sharedMemBytes = 0u;
        kernel_params.kernelParams = kernel_launch_params;
        kernel_params.extra = nullptr;

        CUDA_ERROR_CHECK(cudaGraphAddKernelNode(
            &kernel_node, g, kernel_node_deps, 1, &kernel_params));

        return {kernel_node, a};
    }
};

class alg3 {
    public:
    using result_type = std::monostate;
    struct config_type {
        std::size_t index;
        int *output_ptr;
    };
    using argument_type = std::tuple<int *, std::size_t>;

    static std::tuple<cudaGraphNode_t, result_type> append_graph(
        cudaGraph_t g, cudaGraphNode_t n, config_type c, argument_type a) {
        cudaGraphNode_t allocation_node;
        cudaMemAllocNodeParams alloc_params;
        memset(&alloc_params, 0, sizeof(alloc_params));
        alloc_params.bytesize = sizeof(int);
        alloc_params.poolProps.allocType = cudaMemAllocationTypePinned;
        alloc_params.poolProps.location.id = 0;
        alloc_params.poolProps.location.type = cudaMemLocationTypeDevice;
        CUDA_ERROR_CHECK(cudaGraphAddMemAllocNode(&allocation_node, g, nullptr,
                                                  0, &alloc_params));

        cudaGraphNode_t kernel_node;
        cudaGraphNode_t kernel_node_deps[2] = {allocation_node, n};

        void *kernel_launch_params[3] = {&std::get<0>(a), &alloc_params.dptr,
                                         &c.index};

        cudaKernelNodeParams kernel_params;

#ifdef __CUDACC__
        kernel_params.func = reinterpret_cast<void *>(get_memory_kernel);
#else
        kernel_params.func = nullptr;
#endif

        kernel_params.gridDim = {1u};
        kernel_params.blockDim = {1u};
        kernel_params.sharedMemBytes = 0u;
        kernel_params.kernelParams = kernel_launch_params;
        kernel_params.extra = nullptr;
        CUDA_ERROR_CHECK(cudaGraphAddKernelNode(
            &kernel_node, g, kernel_node_deps, 2, &kernel_params));

        cudaGraphNode_t copy_node;
        cudaGraphNode_t copy_deps[1] = {kernel_node};

        CUDA_ERROR_CHECK(cudaGraphAddMemcpyNode1D(
            &copy_node, g, copy_deps, 1, c.output_ptr, alloc_params.dptr,
            sizeof(int), cudaMemcpyDeviceToHost));

        cudaGraphNode_t free_node;

        CUDA_ERROR_CHECK(cudaGraphAddMemFreeNode(&free_node, g, &copy_node, 1,
                                                 std::get<0>(a)));

        return {copy_node, {}};
    }
};
