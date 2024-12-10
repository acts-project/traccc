/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "get_device.hpp"

#include "hip_error_handling.hpp"

// HIP include(s).
#include <hip/hip_runtime_api.h>

/// @brief Namespace for types that should not be used directly by clients
namespace vecmem::hip::details {

int get_device() {

    int result = 0;
    VECMEM_HIP_ERROR_IGNORE(hipGetDevice(&result));
    return result;
}

}  // namespace vecmem::hip::details
