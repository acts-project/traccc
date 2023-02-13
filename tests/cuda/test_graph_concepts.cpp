/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include "test_graph_algorithms.hpp"
#include "traccc/cuda/graph/graph_descriptor.hpp"

using c1 = traccc::cuda::compose_graphs_initial<alg1, alg2, alg3>;

using c2 = traccc::cuda::compose_graphs<alg2, alg3>;

#ifdef TRACCC_HAVE_CONCEPTS
static_assert(traccc::cuda::graph_descriptor_c<alg1>,
              "`alg1` is not a valid graph descriptor");
static_assert(traccc::cuda::initial_graph_descriptor_c<alg1>,
              "`alg1` is not a valid initial graph descriptor");
static_assert(traccc::cuda::graph_descriptor_c<alg2>,
              "`alg2` is not a valid graph descriptor");
static_assert(traccc::cuda::graph_descriptor_c<alg3>,
              "`alg3` is not a valid graph descriptor");
static_assert(traccc::cuda::graph_descriptor_c<c1>,
              "`c1` is not a valid graph descriptor");
static_assert(traccc::cuda::initial_graph_descriptor_c<c1>,
              "`c1` is not a valid initial graph descriptor");
static_assert(traccc::cuda::graph_descriptor_c<c2>,
              "`c1` is not a valid graph descriptor");
#else
#pragma message "Skipping concept tests."
#endif
