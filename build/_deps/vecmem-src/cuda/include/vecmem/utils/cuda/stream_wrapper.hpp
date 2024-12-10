/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/vecmem_cuda_export.hpp"

// System include(s).
#include <memory>
#include <string>

namespace vecmem {
namespace cuda {

/// @brief Namespace for types that should not be used directly by clients
namespace details {
class opaque_stream;
}

/// Wrapper class for @c cudaStream_t
///
/// It is necessary for passing around CUDA stream objects in code that should
/// not be directly exposed to the CUDA header(s).
///
class stream_wrapper {

public:
    /// Invalid/default device identifier
    static constexpr int INVALID_DEVICE = -1;

    /// Construct a new stream (for the specified device)
    VECMEM_CUDA_EXPORT
    stream_wrapper(int device = INVALID_DEVICE);
    /// Wrap an existing @c cudaStream_t object
    ///
    /// Without taking ownership of it!
    ///
    VECMEM_CUDA_EXPORT
    stream_wrapper(void* stream);

    /// Copy constructor
    VECMEM_CUDA_EXPORT
    stream_wrapper(const stream_wrapper& parent);
    /// Move constructor
    VECMEM_CUDA_EXPORT
    stream_wrapper(stream_wrapper&& parent);

    /// Destructor
    VECMEM_CUDA_EXPORT
    ~stream_wrapper();

    /// Copy assignment
    VECMEM_CUDA_EXPORT
    stream_wrapper& operator=(const stream_wrapper& rhs);
    /// Move assignment
    VECMEM_CUDA_EXPORT
    stream_wrapper& operator=(stream_wrapper&& rhs);

    /// Access a typeless pointer to the managed @c cudaStream_t object
    VECMEM_CUDA_EXPORT
    void* stream() const;

    /// Wait for all queued tasks from the stream to complete
    VECMEM_CUDA_EXPORT
    void synchronize();

private:
    /// Bare pointer to the wrapped @c cudaStream_t object
    void* m_stream;
    /// Smart pointer to the managed @c cudaStream_t object
    std::shared_ptr<details::opaque_stream> m_managedStream;

};  // class stream_wrapper

}  // namespace cuda
}  // namespace vecmem
