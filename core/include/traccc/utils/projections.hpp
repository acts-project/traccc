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
#include "traccc/edm/measurement.hpp"

namespace traccc {

struct [[maybe_unused]] cell_module_projection{
    template <typename CELL_CONTAINER> TRACCC_HOST_DEVICE auto operator()(
        const CELL_CONTAINER& c, std::size_t i)
        const {return c.module_index().at(i);
}
}
;

struct [[maybe_unused]] measurement_module_projection{
    template <typename MEASUREMENT_VECTOR> TRACCC_HOST_DEVICE auto operator()(
        const MEASUREMENT_VECTOR& m, std::size_t i)
        const {return m.at(i).module_link;
}
}
;
}  // namespace traccc
