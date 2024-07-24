/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <traccc/futhark/entry.h>

namespace traccc::futhark {
struct futhark_context& get_context();
}  // namespace traccc::futhark
