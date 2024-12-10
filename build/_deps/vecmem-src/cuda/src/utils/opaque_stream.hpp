/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// CUDA include(s).
#include <cuda_runtime_api.h>

namespace vecmem::cuda::details {

/// RAII wrapper around @c cudaStream_t
///
/// It is used only internally by the VecMem code, so it does not need to
/// provide any nice interface.
///
class opaque_stream {

public:
    /// Default constructor
    opaque_stream();
    /// Destructor
    ~opaque_stream();

    /// Stream managed by the object
    cudaStream_t m_stream;

};  // class opaque_stream

}  // namespace vecmem::cuda::details
