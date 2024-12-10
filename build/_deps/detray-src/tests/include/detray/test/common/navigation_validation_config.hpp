/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// detray plugin include(s)
#include "detray/plugins/svgtools/styling/styling.hpp"

// Detray test include(s).
#include "detray/test/common/detail/whiteboard.hpp"
#include "detray/test/common/fixture_base.hpp"
#include "detray/test/utils/types.hpp"

// System include(s)
#include <limits>
#include <memory>
#include <string>

namespace detray::test {

/// @brief Configuration for a detector scan test.
struct navigation_validation_config
    : public test::fixture_base<>::configuration {
    using base_type = test::fixture_base<>;
    using scalar_type = typename base_type::scalar;
    using vector3_type = typename base_type::vector3;

    /// Name of the test
    std::string m_name{"navigation_validation"};
    /// Access to truth data
    std::shared_ptr<test::whiteboard> m_white_board;
    /// Name of the input file, containing the complete ray scan traces
    std::string m_intersection_file{"truth_intersections.csv"};
    std::string m_track_param_file{"truth_trk_parameters.csv"};
    /// The maximal number of test tracks to run
    std::size_t m_n_tracks{detray::detail::invalid_value<std::size_t>()};
    /// B-field vector for helix
    vector3_type m_B{0.f * unit<scalar_type>::T, 0.f * unit<scalar_type>::T,
                     2.f * unit<scalar_type>::T};
    /// Visualization style to be applied to the svgs
    detray::svgtools::styling::style m_style =
        detray::svgtools::styling::tableau_colorblind::style;

    /// Getters
    /// @{
    const std::string &name() const { return m_name; }
    std::shared_ptr<test::whiteboard> whiteboard() { return m_white_board; }
    std::shared_ptr<test::whiteboard> whiteboard() const {
        return m_white_board;
    }
    const std::string &intersection_file() const { return m_intersection_file; }
    const std::string &track_param_file() const { return m_track_param_file; }
    std::size_t n_tracks() const { return m_n_tracks; }
    const vector3_type &B_vector() { return m_B; }
    const auto &svg_style() const { return m_style; }
    /// @}

    /// Setters
    /// @{
    navigation_validation_config &name(const std::string &n) {
        m_name = n;
        return *this;
    }
    navigation_validation_config &whiteboard(
        std::shared_ptr<test::whiteboard> w_board) {
        if (!w_board) {
            throw std::invalid_argument(
                "Navigation validation: No valid whiteboard instance");
        }
        m_white_board = std::move(w_board);
        return *this;
    }
    navigation_validation_config &intersection_file(const std::string &f) {
        m_intersection_file = f;
        return *this;
    }
    navigation_validation_config &track_param_file(const std::string &f) {
        m_track_param_file = f;
        return *this;
    }
    navigation_validation_config &n_tracks(std::size_t n) {
        m_n_tracks = n;
        return *this;
    }
    navigation_validation_config &B_vector(const vector3_type &B) {
        m_B = B;
        return *this;
    }
    navigation_validation_config &B_vector(const scalar_type B0,
                                           const scalar_type B1,
                                           const scalar_type B2) {
        m_B = vector3_type{B0, B1, B2};
        return *this;
    }
    /// @}
};

}  // namespace detray::test
