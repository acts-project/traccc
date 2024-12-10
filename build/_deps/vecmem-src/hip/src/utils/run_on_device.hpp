/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "hip_error_handling.hpp"
#include "select_device.hpp"

// HIP include(s).
#include <hip/hip_runtime_api.h>

namespace vecmem::hip::details {

/// Helper functor used for running a piece of code on a given device
class run_on_device {

public:
    /// Constructor, with the device that code should run on
    run_on_device(int device) : m_device(device) {}

    /// Operator executing "something" on the specified device.
    template <typename EXECUTABLE>
    void operator()(EXECUTABLE exe) {

        // Switch to the device in a RAII mode.
        select_device helper(m_device);

        // Execute the code.
        exe();
    }

private:
    /// The device to run the code on
    const int m_device;

};  // class run_on_device

}  // namespace vecmem::hip::details
