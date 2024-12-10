/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/units.hpp"
#include "detray/propagator/propagator.hpp"

// Detray test include(s).
#include "detray/test/utils/types.hpp"

// GTest include(s)
#include <gtest/gtest.h>

// System include(s)
#include <limits>
#include <ostream>
#include <string>

namespace detray::test {

/// Base type for test fixtures with google test
template <typename scope = ::testing::Test>
class fixture_base : public scope {
    public:
    /// Linear algebra typedefs
    /// @{
    using algebra_t = ALGEBRA_PLUGIN<test::scalar>;
    using scalar = detray::scalar;
    using point2 = test::point2;
    using point3 = test::point3;
    using vector3 = test::vector3;
    using transform3 = test::transform3;
    /// @}

    /// Local configuration type
    struct configuration {
        /// General testing
        /// @{
        /// Tolerance to compare two floating point values
        float m_tolerance{std::numeric_limits<float>::epsilon()};
        /// Shorthand for infinity
        float inf{std::numeric_limits<float>::infinity()};
        /// Shorthand for the floating point epsilon
        float epsilon{std::numeric_limits<float>::epsilon()};
        /// @}

        /// Propagation
        /// @{
        propagation::config m_prop_cfg{};
        /// @}

        /// Setters
        /// @{
        configuration& tol(float t) {
            m_tolerance = t;
            return *this;
        }
        /// @}

        /// Getters
        /// @{
        float tol() const { return m_tolerance; }
        propagation::config& propagation() { return m_prop_cfg; }
        const propagation::config& propagation() const { return m_prop_cfg; }
        /// @}

        /// Print configuration
        std::ostream& operator<<(std::ostream& os) {
            os << " -> test tolerance:  \t " << tol() << std::endl;
            os << " -> trk path limit:  \t "
               << propagation().stepping.path_limit << std::endl;
            os << " -> overstepping tol:\t "
               << propagation().navigation.overstep_tolerance << std::endl;
            os << " -> step constraint:  \t "
               << propagation().stepping.step_constraint << std::endl;
            os << std::endl;

            return os;
        }
    };

    /// Constructor
    explicit fixture_base(const configuration& cfg = {})
        : tolerance{cfg.tol()},
          inf{cfg.inf},
          epsilon{cfg.epsilon},
          path_limit{cfg.propagation().stepping.path_limit},
          overstep_tolerance{cfg.propagation().navigation.overstep_tolerance},
          step_constraint{cfg.propagation().stepping.step_constraint} {}

    /// @returns the benchmark name
    std::string name() const { return "detray_test"; };

    protected:
    static void SetUpTestSuite() { /* Do nothing */
    }
    static void TearDownTestSuite() { /* Do nothing */
    }
    void SetUp() override { /* Do nothing */
    }
    void TearDown() override { /* Do nothing */
    }

    private:
    float tolerance{};
    float inf{};
    float epsilon{};
    float path_limit{};
    float overstep_tolerance{};
    float step_constraint{};
};

}  // namespace detray::test
