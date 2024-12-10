/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "select_device.hpp"

#include "cuda_error_handling.hpp"

namespace vecmem::cuda::details {
select_device::select_device(int device) {
    /*
     * When the object is constructed, grab the current device number and
     * store it as a member variable. Then set the device to whatever was
     * specified.
     */
    VECMEM_CUDA_ERROR_CHECK(cudaGetDevice(&m_device));
    VECMEM_CUDA_ERROR_CHECK(cudaSetDevice(device));
}

select_device::~select_device() {
    /*
     * On destruction, reset the device number to whatever it was before the
     * object was constructed.
     */
    VECMEM_CUDA_ERROR_CHECK(cudaSetDevice(m_device));
}

int select_device::device() const {

    return m_device;
}

}  // namespace vecmem::cuda::details
