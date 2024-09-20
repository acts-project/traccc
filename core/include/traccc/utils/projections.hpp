/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"

namespace traccc {

struct [[maybe_unused]] measurement_module_projection{
    TRACCC_HOST_DEVICE auto operator()(const traccc::measurement& m)
        const {return m.module_link;
}
}
;
}  // namespace traccc
