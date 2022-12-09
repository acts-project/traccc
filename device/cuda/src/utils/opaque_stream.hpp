/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// CUDA include(s).
#include <cuda_runtime_api.h>

namespace traccc::cuda::details {

/// RAII wrapper around @c cudaStream_t
///
/// It is used only internally by the CUDA library, so it does not need to
/// provide any nice interface.
///
struct opaque_stream {

    /// Default constructor
    opaque_stream();
    /// Destructor
    ~opaque_stream();

    /// Stream managed by the object
    cudaStream_t m_stream;

};  // class opaque_stream

}  // namespace traccc::cuda::details
