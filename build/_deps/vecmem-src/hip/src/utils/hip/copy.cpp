/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include "vecmem/utils/hip/copy.hpp"

#include "../hip_error_handling.hpp"
#include "vecmem/utils/debug.hpp"

// HIP include(s).
#include <hip/hip_runtime_api.h>

// System include(s).
#include <cassert>
#include <string>

namespace vecmem::hip {

/// Helper array for translating between the vecmem and HIP copy type
/// definitions
static constexpr hipMemcpyKind copy_type_translator[copy::type::count] = {
    hipMemcpyHostToDevice, hipMemcpyDeviceToHost, hipMemcpyHostToHost,
    hipMemcpyDeviceToDevice, hipMemcpyDefault};

/// Helper array for providing a printable name for the copy type definitions
static const std::string copy_type_printer[copy::type::count] = {
    "host to device", "device to host", "host to host", "device to device",
    "unknown"};

void copy::do_copy(std::size_t size, const void* from_ptr, void* to_ptr,
                   type::copy_type cptype) const {

    // Check if anything needs to be done.
    if (size == 0) {
        VECMEM_DEBUG_MSG(5, "Skipping unnecessary memory copy");
        return;
    }

    // Some sanity checks.
    assert(from_ptr != nullptr);
    assert(to_ptr != nullptr);
    assert(static_cast<int>(cptype) >= 0);
    assert(static_cast<int>(cptype) < static_cast<int>(copy::type::count));

    // Perform the copy.
    VECMEM_HIP_ERROR_CHECK(
        hipMemcpy(to_ptr, from_ptr, size, copy_type_translator[cptype]));

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(1,
                     "Performed %s memory copy of %lu bytes from %p to "
                     "%p",
                     copy_type_printer[cptype].c_str(), size, from_ptr, to_ptr);
}

void copy::do_memset(std::size_t size, void* ptr, int value) const {

    // Check if anything needs to be done.
    if (size == 0) {
        VECMEM_DEBUG_MSG(5, "Skipping unnecessary memory filling");
        return;
    }

    // Some sanity checks.
    assert(ptr != nullptr);

    // Perform the operation.
    VECMEM_HIP_ERROR_CHECK(hipMemset(ptr, value, size));

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(2, "Set %lu bytes to %i at %p with HIP", size, value, ptr);
}

}  // namespace vecmem::hip
