/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <traccc/futhark/entry.h>

namespace traccc::futhark {
static struct futhark_context_config* __global_futhark_config = nullptr;
static struct futhark_context* __global_futhark_context = nullptr;

struct futhark_context& get_context() {
    if (!__global_futhark_config) {
        __global_futhark_config = futhark_context_config_new();
    }

    if (!__global_futhark_context) {
        __global_futhark_context = futhark_context_new(__global_futhark_config);
    }

    return *__global_futhark_context;
}
}  // namespace traccc::futhark
