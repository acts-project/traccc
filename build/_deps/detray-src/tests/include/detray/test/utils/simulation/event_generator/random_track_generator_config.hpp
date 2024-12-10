/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/math.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/units.hpp"

// Detray test include(s)
#include "detray/test/utils/simulation/event_generator/random_numbers.hpp"

// System include(s)
#include <algorithm>
#include <array>
#include <limits>
#include <ostream>
#include <random>

namespace detray {

/// Configuration for the random track generator
struct random_track_generator_config {

    using seed_t = std::uint64_t;

    /// Gaussian vertex smearing
    bool m_do_vtx_smearing = false;

    /// Monte-Carlo seed
    seed_t m_seed{detail::random_numbers<>::default_seed()};

    /// How many tracks will be generated
    std::size_t m_n_tracks{10u};

    /// Range for phi [-pi, pi) and theta [0, pi)
    std::array<scalar, 2> m_phi_range{-constant<scalar>::pi,
                                      constant<scalar>::pi};
    std::array<scalar, 2> m_theta_range{0.f, constant<scalar>::pi};

    /// Momentum range
    std::array<scalar, 2> m_mom_range{1.f * unit<scalar>::GeV,
                                      1.f * unit<scalar>::GeV};
    /// Whether to interpret the momentum @c m_mom_range as p_T
    bool m_is_pT{false};

    /// Track origin
    std::array<scalar, 3> m_origin{0.f, 0.f, 0.f};
    std::array<scalar, 3> m_origin_stddev{0.f, 0.f, 0.f};

    /// Randomly flip the charge sign?
    bool m_randomize_charge{false};

    /// Time parameter and charge of the track
    scalar m_time{0.f * unit<scalar>::us};
    scalar m_charge{-1.f * unit<scalar>::e};

    /// Setters
    /// @{
    DETRAY_HOST_DEVICE random_track_generator_config& seed(const seed_t s) {
        m_seed = s;
        return *this;
    }
    DETRAY_HOST_DEVICE random_track_generator_config& do_vertex_smearing(
        bool b) {
        m_do_vtx_smearing = b;
        return *this;
    }
    DETRAY_HOST_DEVICE random_track_generator_config& n_tracks(std::size_t n) {
        assert(n > 0);
        m_n_tracks = n;
        return *this;
    }
    DETRAY_HOST_DEVICE random_track_generator_config& phi_range(
        const scalar low, const scalar high) {
        auto min_phi{
            std::clamp(low, -constant<scalar>::pi, constant<scalar>::pi)};
        auto max_phi{
            std::clamp(high, -constant<scalar>::pi, constant<scalar>::pi)};

        assert(min_phi <= max_phi);

        m_phi_range = {min_phi, max_phi};
        return *this;
    }
    template <typename scalar_t>
    DETRAY_HOST_DEVICE random_track_generator_config& phi_range(
        std::array<scalar_t, 2> r) {
        phi_range(static_cast<scalar>(r[0]), static_cast<scalar>(r[1]));
        return *this;
    }
    DETRAY_HOST_DEVICE random_track_generator_config& theta_range(scalar low,
                                                                  scalar high) {
        auto min_theta{std::clamp(low, scalar{0.f}, constant<scalar>::pi)};
        auto max_theta{std::clamp(high, scalar{0.f}, constant<scalar>::pi)};

        assert(min_theta <= max_theta);

        m_theta_range = {min_theta, max_theta};
        return *this;
    }
    template <typename scalar_t>
    DETRAY_HOST_DEVICE random_track_generator_config& theta_range(
        std::array<scalar_t, 2> r) {
        theta_range(static_cast<scalar>(r[0]), static_cast<scalar>(r[1]));
        return *this;
    }
    DETRAY_HOST_DEVICE random_track_generator_config& eta_range(scalar low,
                                                                scalar high) {
        // This value is more or less random
        constexpr auto num_max{0.001f * std::numeric_limits<scalar>::max()};
        auto min_eta{low > -num_max ? low : -num_max};
        auto max_eta{high < num_max ? high : num_max};

        assert(min_eta <= max_eta);

        auto get_theta = [](const scalar eta) {
            return 2.f * math::atan(math::exp(-eta));
        };

        theta_range(get_theta(max_eta), get_theta(min_eta));
        return *this;
    }
    template <typename scalar_t>
    DETRAY_HOST_DEVICE random_track_generator_config& eta_range(
        std::array<scalar_t, 2> r) {
        eta_range(static_cast<scalar>(r[0]), static_cast<scalar>(r[1]));
        return *this;
    }
    DETRAY_HOST_DEVICE random_track_generator_config& mom_range(scalar low,
                                                                scalar high) {
        m_is_pT = false;
        assert(low >= 0.f);
        assert(low <= high);
        m_mom_range = {low, high};
        return *this;
    }
    template <typename scalar_t>
    DETRAY_HOST_DEVICE random_track_generator_config& mom_range(
        std::array<scalar_t, 2> r) {
        mom_range(static_cast<scalar>(r[0]), static_cast<scalar>(r[1]));
        return *this;
    }
    DETRAY_HOST_DEVICE random_track_generator_config& pT_range(scalar low,
                                                               scalar high) {
        m_is_pT = true;
        assert(low >= 0.f);
        assert(low <= high);
        m_mom_range = {low, high};
        return *this;
    }
    template <typename scalar_t>
    DETRAY_HOST_DEVICE random_track_generator_config& pT_range(
        std::array<scalar_t, 2> r) {
        pT_range(static_cast<scalar>(r[0]), static_cast<scalar>(r[1]));
        return *this;
    }
    DETRAY_HOST_DEVICE random_track_generator_config& p_tot(scalar p) {
        mom_range(p, p);
        return *this;
    }
    DETRAY_HOST_DEVICE random_track_generator_config& p_T(scalar p) {
        pT_range(p, p);
        return *this;
    }
    template <typename point3_t = std::array<scalar, 3>>
    DETRAY_HOST_DEVICE random_track_generator_config& origin(point3_t ori) {
        m_origin = {ori[0], ori[1], ori[2]};
        return *this;
    }
    template <typename point3_t = std::array<scalar, 3>>
    DETRAY_HOST_DEVICE random_track_generator_config& origin_stddev(
        point3_t stddev) {
        m_do_vtx_smearing = true;
        m_origin_stddev = {stddev[0], stddev[1], stddev[2]};
        return *this;
    }
    DETRAY_HOST_DEVICE
    random_track_generator_config& randomize_charge(bool rc) {
        m_randomize_charge = rc;
        return *this;
    }
    DETRAY_HOST_DEVICE random_track_generator_config& time(scalar t) {
        assert(t >= 0.f);
        m_time = t;
        return *this;
    }
    DETRAY_HOST_DEVICE random_track_generator_config& charge(scalar q) {
        m_charge = q;
        return *this;
    }
    /// @}

    /// Getters
    /// @{
    DETRAY_HOST_DEVICE constexpr seed_t seed() const { return m_seed; }
    DETRAY_HOST_DEVICE constexpr bool do_vertex_smearing() const {
        return m_do_vtx_smearing;
    }
    DETRAY_HOST_DEVICE constexpr std::size_t n_tracks() const {
        return m_n_tracks;
    }
    DETRAY_HOST_DEVICE constexpr const std::array<scalar, 2>& phi_range()
        const {
        return m_phi_range;
    }
    DETRAY_HOST_DEVICE constexpr const std::array<scalar, 2>& theta_range()
        const {
        return m_theta_range;
    }
    DETRAY_HOST_DEVICE constexpr const std::array<scalar, 2>& mom_range()
        const {
        return m_mom_range;
    }
    DETRAY_HOST_DEVICE constexpr const auto& origin() const { return m_origin; }
    DETRAY_HOST_DEVICE constexpr const auto& origin_stddev() const {
        return m_origin_stddev;
    }
    DETRAY_HOST_DEVICE constexpr bool is_pT() const { return m_is_pT; }
    DETRAY_HOST_DEVICE constexpr bool randomize_charge() const {
        return m_randomize_charge;
    }
    DETRAY_HOST_DEVICE constexpr scalar time() const { return m_time; }
    DETRAY_HOST_DEVICE constexpr scalar charge() const { return m_charge; }
    /// @}

    /// Print the random track generator configuration
    DETRAY_HOST
    friend std::ostream& operator<<(std::ostream& out,
                                    const random_track_generator_config& cfg) {
        const auto& ori = cfg.origin();
        const auto& mom_range = cfg.mom_range();
        const auto& phi_range = cfg.phi_range();
        const auto& theta_range = cfg.theta_range();

        // General
        out << "\nRandom track generator\n"
            << "----------------------------\n"
            << "  No. tracks            : " << cfg.n_tracks() << "\n"
            << "  Charge                : "
            << cfg.charge() / detray::unit<scalar>::e << " [e]\n"
            << "  Rand. charge          : " << std::boolalpha
            << cfg.randomize_charge() << std::noboolalpha << "\n";

        // Momentum and direction
        if (cfg.is_pT()) {
            out << "  Transverse mom.       : [";
        } else {
            out << "  Momentum              : [";
        }
        out << mom_range[0] / detray::unit<scalar>::GeV << ", "
            << mom_range[1] / detray::unit<scalar>::GeV << ") [GeV]\n"
            << "  Phi range             : ["
            << phi_range[0] / detray::unit<scalar>::rad << ", "
            << phi_range[1] / detray::unit<scalar>::rad << ") [rad]\n"
            << "  Theta range           : ["
            << theta_range[0] / detray::unit<scalar>::rad << ", "
            << theta_range[1] / detray::unit<scalar>::rad << ") [rad]\n"
            << "  Origin                : ["
            << ori[0] / detray::unit<scalar>::mm << ", "
            << ori[1] / detray::unit<scalar>::mm << ", "
            << ori[2] / detray::unit<scalar>::mm << "] [mm]\n"
            << "  Do vertex smearing    : " << std::boolalpha
            << cfg.do_vertex_smearing() << "\n"
            << std::noboolalpha;

        if (cfg.do_vertex_smearing()) {
            const auto& ori_stddev = cfg.origin_stddev();
            out << "  Origin stddev         : ["
                << ori_stddev[0] / detray::unit<scalar>::mm << ", "
                << ori_stddev[1] / detray::unit<scalar>::mm << ", "
                << ori_stddev[2] / detray::unit<scalar>::mm << "] [mm]\n";
        }

        return out;
    }
};

}  // namespace detray
