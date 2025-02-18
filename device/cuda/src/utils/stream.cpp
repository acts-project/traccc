/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/cuda/utils/stream.hpp"

#include "cuda_error_handling.hpp"
#include "opaque_stream.hpp"
#include "utils.hpp"

#include <cuda_runtime_api.h>

namespace traccc::cuda {

stream::stream(int device) {

    // Make sure that the stream is constructed on the correct device.
    details::select_device dev_selector{
        device == INVALID_DEVICE ? details::get_device() : device};

    // Construct the stream.
    m_stream = std::make_unique<details::opaque_stream>(dev_selector.device());
}

stream::stream(stream&& parent) : m_stream(std::move(parent.m_stream)) {}

/// The destructor is implemented explicitly to avoid clients of the class
/// having to know how to destruct @c traccc::cuda::details::opaque_stream.
stream::~stream() {}

stream& stream::operator=(stream&& rhs) {

    // Avoid self-assignment.
    if (this == &rhs) {
        return *this;
    }

    // Move the managed queue object.
    m_stream = std::move(rhs.m_stream);

    // Return this object.
    return *this;
}

int stream::device() const {

    return m_stream->m_device;
}

void* stream::cudaStream() const {

    return m_stream->m_stream;
}

void stream::synchronize() const {

    TRACCC_CUDA_ERROR_CHECK(cudaStreamSynchronize(m_stream->m_stream));
}

}  // namespace traccc::cuda
