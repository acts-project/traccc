/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem::hip::details {

/// Helper function for determining the "currently active device"
///
/// Note that calling the function on a machine with no HIP device does not
/// result in an error, the function just returns 0 in that case.
///
int get_device();

}  // namespace vecmem::hip::details
