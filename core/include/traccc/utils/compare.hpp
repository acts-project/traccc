/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

namespace traccc {

template <template <typename, typename> class pair_t, typename T>
struct compare_pair_int {
    TRACCC_HOST_DEVICE
    bool operator()(const pair_t<T, T> &a, const T &b) { return (a.first < b); }
    TRACCC_HOST_DEVICE
    bool operator()(const T &a, const pair_t<T, T> &b) { return (a < b.first); }
};

}  // namespace traccc