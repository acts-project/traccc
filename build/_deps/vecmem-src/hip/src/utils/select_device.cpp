/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "select_device.hpp"

#include "get_device.hpp"
#include "hip_error_handling.hpp"

// HIP include(s).
#include <hip/hip_runtime_api.h>

namespace vecmem::hip::details {

select_device::select_device(int device) : m_device(get_device()) {

    VECMEM_HIP_ERROR_CHECK(hipSetDevice(device));
}

select_device::~select_device() {

    VECMEM_HIP_ERROR_CHECK(hipSetDevice(m_device));
}

}  // namespace vecmem::hip::details
