#pragma once

#include "edm/cell.hpp"
#include "edm/measurement.hpp"

#include "vecmem/memory/memory_resource.hpp"

#include <vector>

#define MAX_ACTIVATIONS_PER_MODULE 2048
#define MAX_CLUSTERS_PER_MODULE 128
#define THREADS_PER_BLOCK 128

namespace traccc::cuda::details {
    void
    sparse_ccl(
        const cell * _cells,
        const unsigned int * blocks,
        float * _out,
        std::size_t modules
    );
}
