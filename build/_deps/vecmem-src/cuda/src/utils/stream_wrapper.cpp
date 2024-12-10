/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/utils/cuda/stream_wrapper.hpp"

#include "cuda_error_handling.hpp"
#include "cuda_wrappers.hpp"
#include "get_device_name.hpp"
#include "opaque_stream.hpp"
#include "select_device.hpp"

// VecMem include(s).
#include "vecmem/utils/debug.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

namespace vecmem::cuda {

stream_wrapper::stream_wrapper(int device)
    : m_stream(nullptr), m_managedStream() {

    // Make sure that the stream is constructed on the correct device.
    details::select_device dev_selector(
        device == INVALID_DEVICE ? details::get_device() : device);

    // Construct the stream.
    m_managedStream = std::make_shared<details::opaque_stream>();
    m_stream = m_managedStream->m_stream;

    // Tell the user what happened.
    VECMEM_DEBUG_MSG(1, "Created stream on device: %s",
                     details::get_device_name(dev_selector.device()).c_str());
}

stream_wrapper::stream_wrapper(void* stream)
    : m_stream(stream), m_managedStream() {}

stream_wrapper::stream_wrapper(const stream_wrapper& parent)
    : m_stream(parent.m_stream), m_managedStream(parent.m_managedStream) {}

stream_wrapper::stream_wrapper(stream_wrapper&& parent)
    : m_stream(parent.m_stream),
      m_managedStream(std::move(parent.m_managedStream)) {}

stream_wrapper::~stream_wrapper() {}

stream_wrapper& stream_wrapper::operator=(const stream_wrapper& rhs) {

    // Avoid self-assignment.
    if (this == &rhs) {
        return *this;
    }

    // Copy the stream.
    m_stream = rhs.m_stream;
    m_managedStream = rhs.m_managedStream;

    // Return this object.
    return *this;
}

stream_wrapper& stream_wrapper::operator=(stream_wrapper&& rhs) {

    // Avoid self-assignment.
    if (this == &rhs) {
        return *this;
    }

    // Move the managed queue object, and copy the pointer.
    m_stream = rhs.m_stream;
    m_managedStream = std::move(rhs.m_managedStream);

    // Return this object.
    return *this;
}

void* stream_wrapper::stream() const {

    return m_stream;
}

void stream_wrapper::synchronize() {

    VECMEM_CUDA_ERROR_CHECK(cudaStreamSynchronize(details::get_stream(*this)));
}

}  // namespace vecmem::cuda
