/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cell.hpp"

namespace traccc {
struct [[maybe_unused]] channel0_major_cell_order_relation{
    TRACCC_HOST_DEVICE [[maybe_unused]] bool operator()(const traccc::cell& a,
                                                        const traccc::cell& b)
        const {if (a.module_link ==
                   b.module_link){if (a.channel1 <= b.channel1){return true;
}
else if (a.channel1 == b.channel1) {
    return a.channel0 <= b.channel0;
}
else {
    return false;
}
}
else {
    return true;
}
}
}
;
}  // namespace traccc
