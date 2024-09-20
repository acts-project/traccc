/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

namespace traccc {
struct [[maybe_unused]] channel0_major_cell_order_relation{
    template <typename CELL_CONTAINER> TRACCC_HOST_DEVICE [[maybe_unused]] bool
    operator()(const CELL_CONTAINER& c, std::size_t i, std::size_t j)
        const {if (c.module_index().at(i) == c.module_index().at(j)){
            if (c.channel1().at(i) == c.channel1().at(j)){
                return c.channel0().at(i) <= c.channel0().at(j);
}
else {
    return c.channel1().at(i) <= c.channel1().at(j);
}
}
else {
    return true;
}
}
}
;
}  // namespace traccc
