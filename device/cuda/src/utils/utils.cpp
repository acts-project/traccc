/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "utils.hpp"

#include "cuda_error_handling.hpp"

namespace traccc::cuda::details {

int get_device() {

    int d = -1;
    cudaGetDevice(&d);
    return d;
}

unsigned int get_warp_size(int device) {

    int warp_size = 0;
    TRACCC_CUDA_ERROR_CHECK(
        cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device));
    return static_cast<unsigned int>(warp_size);
}

cudaStream_t get_stream(const stream& stream) {

    return reinterpret_cast<cudaStream_t>(stream.cudaStream());
}

select_device::select_device(int device) {
    /*
     * When the object is constructed, grab the current device number and
     * store it as a member variable. Then set the device to whatever was
     * specified.
     */
    TRACCC_CUDA_ERROR_CHECK(cudaGetDevice(&m_device));
    TRACCC_CUDA_ERROR_CHECK(cudaSetDevice(device));
}

select_device::~select_device() {
    /*
     * On destruction, reset the device number to whatever it was before the
     * object was constructed.
     */
    TRACCC_CUDA_ERROR_CHECK(cudaSetDevice(m_device));
}

int select_device::device() const {

    return m_device;
}

}  // namespace traccc::cuda::details
