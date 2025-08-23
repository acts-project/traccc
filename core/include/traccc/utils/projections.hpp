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
#include "traccc/edm/silicon_cell_collection.hpp"

namespace traccc {

struct [[maybe_unused]] cell_module_projection {
    template <typename T>
    TRACCC_HOST_DEVICE auto operator()(const edm::silicon_cell<T>& c) const {
        return c.module_index();
    }
};

struct [[maybe_unused]] measurement_module_projection {
    TRACCC_HOST_DEVICE auto operator()(const traccc::measurement& m) const {
        return m.surface_link;
    }
};
}  // namespace traccc
