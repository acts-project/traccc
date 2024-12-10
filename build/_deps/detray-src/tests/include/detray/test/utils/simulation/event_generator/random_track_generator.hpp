/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/math.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/units.hpp"
#include "detray/utils/ranges/ranges.hpp"

// Detray test include(s)
#include "detray/test/utils/simulation/event_generator/random_track_generator_config.hpp"

// System include(s)
#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <random>

namespace detray {

/// @brief Generates track states with random momentum directions.
///
/// Generates the phi and theta angles of the track momentum according to a
/// given random number distribution.
///
/// @tparam track_t the type of track parametrization that should be used.
/// @tparam generator_t source of random numbers
///
/// @note Since the random number generator might not be copy constructible,
/// neither is this generator. The iterators hold a reference to the rand
/// generator, which must not be invalidated during the iteration.
/// @note the random numbers are clamped to fit the phi/theta ranges. This can
/// effect distribution mean etc.
template <typename track_t, typename generator_t = detail::random_numbers<>>
class random_track_generator
    : public detray::ranges::view_interface<
          random_track_generator<track_t, generator_t>> {

    using point3 = typename track_t::point3_type;
    using vector3 = typename track_t::vector3_type;

    public:
    using track_type = track_t;

    /// Configure how tracks are generated
    using configuration = random_track_generator_config;

    private:
    /// @brief Nested iterator type that generates track states.
    struct iterator {

        using difference_type = std::ptrdiff_t;
        using value_type = track_t;
        using pointer = track_t*;
        using reference = track_t&;
        using iterator_category = detray::ranges::input_iterator_tag;

        constexpr iterator() = delete;

        DETRAY_HOST_DEVICE
        iterator(std::shared_ptr<generator_t> rand_gen, configuration cfg,
                 std::size_t n_tracks)
            : m_rnd_numbers{std::move(rand_gen)},
              m_tracks{n_tracks},
              m_cfg{cfg} {}

        /// @returns whether we reached the end of iteration
        DETRAY_HOST_DEVICE
        constexpr bool operator==(const iterator& rhs) const {
            return rhs.m_tracks == m_tracks;
        }

        /// @returns the generator at its next position (prefix)
        DETRAY_HOST_DEVICE
        constexpr auto operator++() -> iterator& {
            ++m_tracks;
            return *this;
        }

        /// @returns the generator at its next position (postfix)
        DETRAY_HOST_DEVICE
        constexpr auto operator++(int) -> iterator& {
            auto tmp(m_tracks);
            ++m_tracks;
            return tmp;
        }

        /// @returns a track instance from random-generated momentum
        DETRAY_HOST_DEVICE
        track_t operator*() const {

            if (!m_rnd_numbers) {
                throw std::invalid_argument("Invalid random number generator");
            }

            const auto& ori = m_cfg.origin();
            const auto& ori_stddev = m_cfg.origin_stddev();

            const point3 vtx =
                m_cfg.do_vertex_smearing()
                    ? point3{m_rnd_numbers->normal(ori[0], ori_stddev[0]),
                             m_rnd_numbers->normal(ori[1], ori_stddev[1]),
                             m_rnd_numbers->normal(ori[2], ori_stddev[2])}
                    : point3{ori[0], ori[1], ori[2]};

            scalar p_mag{(*m_rnd_numbers)(m_cfg.mom_range())};
            scalar phi{(*m_rnd_numbers)(m_cfg.phi_range())};
            scalar theta{(*m_rnd_numbers)(m_cfg.theta_range())};
            scalar sin_theta{math::sin(theta)};

            // Momentum direction from angles
            vector3 mom{math::cos(phi) * sin_theta, math::sin(phi) * sin_theta,
                        math::cos(theta)};

            sin_theta = (sin_theta == scalar{0.f})
                            ? std::numeric_limits<scalar>::epsilon()
                            : sin_theta;

            mom = (m_cfg.is_pT() ? 1.f / sin_theta : 1.f) * p_mag *
                  vector::normalize(mom);

            // Randomly flip the charge sign
            std::array<double, 2> signs{1., -1.};
            const auto sign{static_cast<scalar>(
                signs[m_cfg.randomize_charge() ? m_rnd_numbers->coin_toss()
                                               : 0u])};

            return track_t{vtx, m_cfg.time(), mom, sign * m_cfg.charge()};
        }

        /// Random number generator
        std::shared_ptr<generator_t> m_rnd_numbers;

        /// How many tracks will be generated
        std::size_t m_tracks{0u};

        /// Configuration
        configuration m_cfg{};
    };

    std::shared_ptr<generator_t> m_gen{nullptr};
    configuration m_cfg{};

    public:
    using iterator_t = iterator;

    /// Default constructor
    constexpr random_track_generator() = default;

    /// Construct from external configuration
    DETRAY_HOST_DEVICE
    explicit constexpr random_track_generator(const configuration& cfg)
        : m_gen{std::make_shared<generator_t>(cfg.seed())}, m_cfg(cfg) {}

    /// Paramtetrized constructor for quick construction of simple tasks
    ///
    /// @note For more complex tasks, use the @c configuration type
    ///
    /// @param n_tracks the number of steps in the theta space
    /// @param mom_range the range of the track momentum (in GeV)
    /// @param charge charge of particle (e)
    DETRAY_HOST_DEVICE
    random_track_generator(
        std::size_t n_tracks,
        std::array<scalar, 2> mom_range = {1.f * unit<scalar>::GeV,
                                           1.f * unit<scalar>::GeV},
        scalar charge = -1.f * unit<scalar>::e)
        : m_gen{std::make_shared<generator_t>()}, m_cfg{} {
        m_cfg.n_tracks(n_tracks);
        m_cfg.mom_range(mom_range);
        m_cfg.charge(charge);
    }

    /// Move constructor
    DETRAY_HOST_DEVICE
    random_track_generator(random_track_generator&& other) noexcept
        : m_gen(std::move(other.m_gen)), m_cfg(std::move(other.m_cfg)) {}

    /// Access the configuration
    DETRAY_HOST_DEVICE
    constexpr configuration& config() { return m_cfg; }

    /// @returns the generator in initial state.
    /// @note the underlying random number generator has deleted copy
    /// constructor, so the iterator needs to be built from scratch
    DETRAY_HOST_DEVICE
    auto begin() noexcept -> iterator { return {m_gen, m_cfg, 0u}; }

    /// @returns the generator in end state
    DETRAY_HOST_DEVICE
    auto end() noexcept -> iterator { return {m_gen, m_cfg, m_cfg.n_tracks()}; }

    /// @returns the number of tracks that will be generated
    DETRAY_HOST_DEVICE
    constexpr auto size() const noexcept -> std::size_t {
        return m_cfg.n_tracks();
    }
};

}  // namespace detray
