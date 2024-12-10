/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/units.hpp"

// Detray test include(s)
#include "detray/test/utils/simulation/event_generator/random_numbers.hpp"

// System include(s)
#include <algorithm>
#include <array>
#include <limits>
#include <ostream>

namespace detray {

/// Configuration for the uniform track generator
struct uniform_track_generator_config {

    using seed_t = std::uint64_t;

    /// Monte-Carlo seed
    seed_t m_seed{detail::random_numbers<>::default_seed()};

    /// Ensure same angle space as random track generator
    static constexpr scalar k_max_pi{constant<scalar>::pi -
                                     std::numeric_limits<scalar>::epsilon()};

    /// Range for phi [-pi, pi) and theta [0, pi)
    std::array<scalar, 2> m_phi_range{-constant<scalar>::pi, k_max_pi};
    std::array<scalar, 2> m_theta_range{0.f, k_max_pi};
    std::array<scalar, 2> m_eta_range{-5.f, 5.f};

    /// Angular step size
    std::size_t m_phi_steps{50u};
    std::size_t m_theta_steps{50u};

    /// Do uniform eta steps instead of uniform theta steps
    /// (use same number of steps and range)
    bool m_uniform_eta{false};

    /// Track origin
    std::array<scalar, 3> m_origin{0.f, 0.f, 0.f};

    /// Magnitude of momentum: Default is one to keep directions normalized
    /// if no momentum information is needed (e.g. for a ray)
    scalar m_p_mag{1.f * unit<scalar>::GeV};
    /// Whether to interpret the momentum @c m_p_mag as p_T
    bool m_is_pT{false};

    /// Randomly flip the charge sign?
    bool m_randomize_charge{false};

    /// Time parameter and charge of the track
    scalar m_time{0.f * unit<scalar>::us};
    scalar m_charge{-1.f * unit<scalar>::e};

    /// Setters
    /// @{
    DETRAY_HOST_DEVICE uniform_track_generator_config& seed(const seed_t s) {
        m_seed = s;
        return *this;
    }
    DETRAY_HOST_DEVICE uniform_track_generator_config& phi_range(scalar low,
                                                                 scalar high) {
        auto min_phi{std::clamp(low, -constant<scalar>::pi, k_max_pi)};
        auto max_phi{std::clamp(high, -constant<scalar>::pi, k_max_pi)};

        assert(min_phi <= max_phi);

        m_phi_range = {min_phi, max_phi};
        return *this;
    }
    template <typename scalar_t>
    DETRAY_HOST_DEVICE uniform_track_generator_config& phi_range(
        std::array<scalar_t, 2> r) {
        phi_range(static_cast<scalar>(r[0]), static_cast<scalar>(r[1]));
        return *this;
    }
    DETRAY_HOST_DEVICE uniform_track_generator_config& theta_range(
        scalar low, scalar high) {
        auto min_theta{std::clamp(low, scalar{0.f}, k_max_pi)};
        auto max_theta{std::clamp(high, scalar{0.f}, k_max_pi)};

        assert(min_theta <= max_theta);

        m_theta_range = {min_theta, max_theta};
        m_uniform_eta = false;
        return *this;
    }
    template <typename scalar_t>
    DETRAY_HOST_DEVICE uniform_track_generator_config& theta_range(
        std::array<scalar_t, 2> r) {
        theta_range(static_cast<scalar>(r[0]), static_cast<scalar>(r[1]));
        return *this;
    }
    DETRAY_HOST_DEVICE uniform_track_generator_config& eta_range(scalar low,
                                                                 scalar high) {
        // This value is more or less random
        constexpr auto num_max{0.001f * std::numeric_limits<scalar>::max()};
        auto min_eta{low > -num_max ? low : -num_max};
        auto max_eta{high < num_max ? high : num_max};

        assert(min_eta <= max_eta);

        m_eta_range = {min_eta, max_eta};
        m_uniform_eta = true;
        return *this;
    }
    template <typename scalar_t>
    DETRAY_HOST_DEVICE uniform_track_generator_config& eta_range(
        std::array<scalar_t, 2> r) {
        eta_range(static_cast<scalar>(r[0]), static_cast<scalar>(r[1]));
        return *this;
    }
    DETRAY_HOST_DEVICE uniform_track_generator_config& phi_steps(
        std::size_t n) {
        assert(n > 0);
        m_phi_steps = n;
        return *this;
    }
    DETRAY_HOST_DEVICE uniform_track_generator_config& theta_steps(
        std::size_t n) {
        assert(n > 0);
        m_theta_steps = n;
        m_uniform_eta = false;
        return *this;
    }
    DETRAY_HOST_DEVICE uniform_track_generator_config& eta_steps(
        std::size_t n) {
        assert(n > 0);
        m_theta_steps = n;
        m_uniform_eta = true;
        return *this;
    }
    DETRAY_HOST_DEVICE uniform_track_generator_config& uniform_eta(bool b) {
        m_uniform_eta = b;
        return *this;
    }
    template <typename point3_t = std::array<scalar, 3>>
    DETRAY_HOST_DEVICE uniform_track_generator_config& origin(point3_t ori) {
        m_origin = {ori[0], ori[1], ori[2]};
        return *this;
    }
    DETRAY_HOST_DEVICE uniform_track_generator_config& p_tot(scalar p) {
        assert(p > 0.f);
        m_is_pT = false;
        m_p_mag = p;
        return *this;
    }
    DETRAY_HOST_DEVICE uniform_track_generator_config& p_T(scalar p) {
        assert(p > 0.f);
        m_is_pT = true;
        m_p_mag = p;
        return *this;
    }
    DETRAY_HOST_DEVICE
    uniform_track_generator_config& randomize_charge(bool rc) {
        m_randomize_charge = rc;
        return *this;
    }
    DETRAY_HOST_DEVICE uniform_track_generator_config& time(scalar t) {
        m_time = t;
        return *this;
    }
    DETRAY_HOST_DEVICE uniform_track_generator_config& charge(scalar q) {
        m_charge = q;
        return *this;
    }
    /// @}

    /// Getters
    /// @{
    DETRAY_HOST_DEVICE constexpr seed_t seed() const { return m_seed; }
    DETRAY_HOST_DEVICE constexpr std::size_t n_tracks() const {
        return phi_steps() * theta_steps();
    }
    DETRAY_HOST_DEVICE constexpr std::array<scalar, 2> phi_range() const {
        return m_phi_range;
    }
    DETRAY_HOST_DEVICE constexpr std::array<scalar, 2> theta_range() const {
        return m_theta_range;
    }
    DETRAY_HOST_DEVICE constexpr std::array<scalar, 2> eta_range() const {
        return m_eta_range;
    }
    DETRAY_HOST_DEVICE constexpr std::size_t phi_steps() const {
        return m_phi_steps;
    }
    DETRAY_HOST_DEVICE constexpr std::size_t theta_steps() const {
        return m_theta_steps;
    }
    DETRAY_HOST_DEVICE constexpr std::size_t eta_steps() const {
        return m_theta_steps;
    }
    DETRAY_HOST_DEVICE constexpr bool uniform_eta() const {
        return m_uniform_eta;
    }
    DETRAY_HOST_DEVICE constexpr const auto& origin() const { return m_origin; }
    DETRAY_HOST_DEVICE constexpr bool is_pT() const { return m_is_pT; }
    DETRAY_HOST_DEVICE constexpr bool randomize_charge() const {
        return m_randomize_charge;
    }
    DETRAY_HOST_DEVICE constexpr scalar time() const { return m_time; }
    DETRAY_HOST_DEVICE constexpr scalar charge() const { return m_charge; }
    /// @}

    /// Print the unifrom track generator configuration
    DETRAY_HOST
    friend std::ostream& operator<<(std::ostream& out,
                                    const uniform_track_generator_config& cfg) {
        const auto& ori = cfg.origin();
        const auto& phi_range = cfg.phi_range();

        // General
        out << "\nUnform track generator\n"
            << "----------------------------\n"
            << "  No. tracks            : " << cfg.n_tracks() << "\n"
            << "    -> phi steps        : " << cfg.phi_steps() << "\n"
            << "    -> theta/eta steps  : " << cfg.theta_steps() << "\n"
            << "  Charge                : "
            << cfg.charge() / detray::unit<scalar>::e << " [e]\n"
            << "  Rand. charge          : " << std::boolalpha
            << cfg.randomize_charge() << std::noboolalpha << "\n";

        // Momentum
        if (cfg.is_pT()) {
            out << "  Transverse mom.       : "
                << cfg.m_p_mag / detray::unit<scalar>::GeV << " [GeV]\n";
        } else {
            out << "  Momentum              : "
                << cfg.m_p_mag / detray::unit<scalar>::GeV << " [GeV]\n";
        }

        // Direction
        out << "  Phi range             : ["
            << phi_range[0] / detray::unit<scalar>::rad << ", "
            << phi_range[1] / detray::unit<scalar>::rad << ") [rad]\n";
        if (cfg.uniform_eta()) {
            const auto& eta_range = cfg.eta_range();
            out << "  Eta range             : [" << eta_range[0] << ", "
                << eta_range[1] << "]\n";
        } else {
            const auto& theta_range = cfg.theta_range();
            out << "  Theta range           : ["
                << theta_range[0] / detray::unit<scalar>::rad << ", "
                << theta_range[1] / detray::unit<scalar>::rad << ") [rad]\n";
        }

        // Origin
        out << "  Origin                : ["
            << ori[0] / detray::unit<scalar>::mm << ", "
            << ori[1] / detray::unit<scalar>::mm << ", "
            << ori[2] / detray::unit<scalar>::mm << "] [mm]\n";

        return out;
    }
};

}  // namespace detray
