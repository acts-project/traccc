/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cuda_runtime.h>

#include "traccc/cuda/graph/graph_descriptor.hpp"
#include "traccc/cuda/utils/definitions.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc::cuda {
template <CONSTRAINT(initial_graph_descriptor_c) G>
class graph_algorithm
    : public algorithm<typename G::result_type(
          cudaStream_t, typename G::config_type, typename G::argument_type)> {
    public:
    graph_algorithm() {}

    ~graph_algorithm() {}

    virtual typename G::result_type operator()(
        cudaStream_t s, typename G::config_type config,
        typename G::argument_type args) const override {
        cudaGraph_t g;
        typename G::result_type r;

        std::tie(g, std::ignore, r) = G::create_graph(config, args);

        cudaGraphExec_t e;

        CUDA_ERROR_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));

        CUDA_ERROR_CHECK(cudaGraphLaunch(e, s));

        CUDA_ERROR_CHECK(cudaStreamSynchronize(s));

        CUDA_ERROR_CHECK(cudaGraphExecDestroy(e));

        CUDA_ERROR_CHECK(cudaGraphDestroy(g));

        return r;
    }
};
}  // namespace traccc::cuda
