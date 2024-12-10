/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// detray definitions
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/containers.hpp"
#include "detray/definitions/detail/math.hpp"
#include "detray/definitions/detail/qualifiers.hpp"

// System include(s).
#include <limits>
#include <type_traits>

namespace detray {

// Add constraints for steppers
namespace step {

/// Direction in which the integration is performed
enum class direction : int {
    e_forward = 1,
    e_unknown = std::numeric_limits<int>::max(),
    e_backward = -1,
};

/// the types of constraints
/// from accuracy - this can vary up and down given a good step estimator
/// from actor    - this would be a typical navigation step
/// from aborter  - this would be a target condition
/// from user     - this is user given for what reason ever
enum constraint : std::size_t {
    e_accuracy = 0u,
    e_actor = 1u,
    e_aborter = 2u,
    e_user = 3u,
    e_all = 4u
};

}  // namespace step

/// Struct that represents unconstrained stepping
struct unconstrained_step {

    /// Register a new @param step_size constraint
    template <step::constraint type>
    DETRAY_HOST_DEVICE constexpr void set(const scalar /*step_size*/) const {
        /*Do nothing*/
    }

    /// @returns the current step size constraint
    template <step::constraint type = step::constraint::e_all>
    DETRAY_HOST_DEVICE constexpr scalar size(
        const step::direction /*dir*/ = step::direction::e_forward) const {
        return std::numeric_limits<scalar>::max();
    }

    /// Remove constraints
    template <step::constraint type = step::constraint::e_actor>
    DETRAY_HOST_DEVICE constexpr void release() const {
        /*Do nothing*/
    }
};

/// Struct that can be configured with a number of different step sizes by other
/// actors and will then resolve the strictest one.
template <template <typename, std::size_t> class array_t = darray>
struct constrained_step {

    /// Register a new @param step_size constraint
    template <step::constraint type>
    requires(type != step::constraint::e_all) DETRAY_HOST_DEVICE
        void set(const scalar step_size) {
        _constraints[type] =
            math::min(_constraints[type], math::fabs(step_size));
    }

    /// @returns the current step size constraint for a given type or overall
    template <step::constraint type = step::constraint::e_all>
    DETRAY_HOST_DEVICE scalar
    size(const step::direction dir = step::direction::e_forward) const {
        if constexpr (type == step::constraint::e_all) {
            return static_cast<scalar>(dir) * min();
        } else {
            return static_cast<scalar>(dir) * _constraints[type];
        }
    }

    /// Remove [all] constraints
    template <step::constraint type = step::constraint::e_actor>
    DETRAY_HOST_DEVICE void release() {
        if constexpr (type == step::constraint::e_all) {
            _constraints = {std::numeric_limits<scalar>::max(),
                            std::numeric_limits<scalar>::max(),
                            std::numeric_limits<scalar>::max(),
                            std::numeric_limits<scalar>::max()};
        } else {
            _constraints[type] = std::numeric_limits<scalar>::max();
        }
    }

    /// @returns the strongest constraint
    DETRAY_HOST_DEVICE scalar min() const {
        scalar min_constr = std::numeric_limits<scalar>::max();
        min_constr =
            math::min(min_constr, _constraints[step::constraint::e_accuracy]);
        min_constr =
            math::min(min_constr, _constraints[step::constraint::e_actor]);
        min_constr =
            math::min(min_constr, _constraints[step::constraint::e_aborter]);
        return math::min(min_constr, _constraints[step::constraint::e_user]);
    }

    /// Current step size constraints from accuracy, actors, aborters or user
    array_t<scalar, 4> _constraints = {
        std::numeric_limits<scalar>::max(), std::numeric_limits<scalar>::max(),
        std::numeric_limits<scalar>::max(), std::numeric_limits<scalar>::max()};
};

}  // namespace detray
