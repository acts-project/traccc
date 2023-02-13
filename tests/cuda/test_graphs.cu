/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include "test_graph_algorithms.hpp"
#include "traccc/cuda/graph/graph_algorithm.hpp"
#include "traccc/cuda/graph/graph_descriptor.hpp"
#include "traccc/cuda/utils/definitions.hpp"

using c = traccc::cuda::compose_graphs_initial<alg1, alg2, alg3>;

TEST(CUDAGraphBasic, LaunchGraph) {
    traccc::cuda::graph_algorithm<c> g;

    cudaStream_t s;

    CUDA_ERROR_CHECK(cudaStreamCreate(&s));

    int result;

    g(s,
      {alg1::config_type{1024}, alg2::config_type{},
       alg3::config_type{5, &result}},
      {});

    CUDA_ERROR_CHECK(cudaStreamDestroy(s));

    ASSERT_EQ(result, 5);
}
