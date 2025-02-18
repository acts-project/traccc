/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/performance/details/is_same_object.hpp"

namespace traccc::details {

/// Factory creating instances of "comparator objects" for a given type
///
/// This level of abstraction is necessary to be able to construct comparator
/// objects that would have extra configuration parameters over the reference
/// object and the comparison uncertainty.
///
/// @tparam TYPE The type for which a comparator object should be generated
///
template <typename TYPE>
struct projector {
    static constexpr bool exists = false;
};

template <>
struct projector<traccc::measurement> {
    static constexpr bool exists = true;

    float operator()(const traccc::measurement& i) {
        return static_cast<float>(i.local[0]);
    }
};

template <>
struct projector<traccc::spacepoint> {
    static constexpr bool exists = true;

    float operator()(const traccc::spacepoint& i) {
        return static_cast<float>(i.x());
    }
};

template <>
struct projector<traccc::seed> {
    static constexpr bool exists = true;

    float operator()(const traccc::seed& i) {
        return static_cast<float>(i.z_vertex);
    }
};

template <>
struct projector<traccc::bound_track_parameters> {
    static constexpr bool exists = true;

    float operator()(const traccc::bound_track_parameters& i) {
        return static_cast<float>(i.phi());
    }
};

template <>
struct projector<fitting_result<traccc::default_algebra>> {
    static constexpr bool exists = true;

    float operator()(const fitting_result<traccc::default_algebra>& i) {
        return static_cast<float>(i.ndf);
    }
};
}  // namespace traccc::details
