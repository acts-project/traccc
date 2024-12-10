/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/static_array.hpp"
#include "vecmem/utils/cuda/stream_wrapper.hpp"

/// Perform a linear transformation using the received vectors
void linearTransform(vecmem::data::vector_view<const int> constants,
                     vecmem::data::vector_view<const int> input,
                     vecmem::data::vector_view<int> output);

/// Perform an asynchronous linear transformation using the received vectors
void linearTransform(vecmem::data::vector_view<const int> constants,
                     vecmem::data::vector_view<const int> input,
                     vecmem::data::vector_view<int> output,
                     const vecmem::cuda::stream_wrapper& stream);

/// Function incrementing the elements of the received vector using atomics
void atomicTransform(unsigned int iterations,
                     vecmem::data::vector_view<int> vec);

/// Function filling vectors after using atomics in local address space
void atomicLocalRef(unsigned int num_blocks, unsigned int block_size,
                    vecmem::data::vector_view<int> vec);

/// Function filtering elements of an input vector into an output vector
void filterTransform(vecmem::data::vector_view<const int> input,
                     vecmem::data::vector_view<int> output);

/// Function filtering elements of an input vector into an output vector
void filterTransform(vecmem::data::jagged_vector_view<const int> input,
                     unsigned int max_vec_size,
                     vecmem::data::jagged_vector_view<int> output);

/// Function filling the jagged vector to its capacity
void fillTransform(vecmem::data::jagged_vector_view<int> vec);

/// Function transforming the elements of an array of vectors
void arrayTransform(
    vecmem::static_array<vecmem::data::vector_view<int>, 4> data);

/// Function performing a trivial operation on a "large" vector buffer
void largeBufferTransform(vecmem::data::vector_view<unsigned long> data);

/// Function performing a trivial operation on a "large" jagged vector buffer
void largeBufferTransform(vecmem::data::jagged_vector_view<unsigned long> data);
