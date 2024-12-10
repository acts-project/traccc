/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// TODO: Remove this when gcc fixes their false positives.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic warning "-Wmaybe-uninitialized"
#endif

// Project include(s)
#include "detray/geometry/tracking_surface.hpp"
#include "detray/navigation/detail/ray.hpp"
#include "detray/navigation/navigation_config.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/base_actor.hpp"
#include "detray/propagator/base_stepper.hpp"
#include "detray/propagator/stepping_config.hpp"
#include "detray/utils/tuple_helpers.hpp"

// System include(s)
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace detray {

/// An inspector that aggregates a number of different inspectors.
template <typename... Inspectors>
struct aggregate_inspector {

    using view_type = dmulti_view<detail::get_view_t<Inspectors>...>;
    using const_view_type =
        dmulti_view<detail::get_view_t<const Inspectors>...>;

    using inspector_tuple_t = std::tuple<Inspectors...>;
    inspector_tuple_t _inspectors{};

    /// Default constructor
    aggregate_inspector() = default;

    /// Construct from the inspector @param view type. Mainly used device-side.
    template <concepts::device_view view_t>
    DETRAY_HOST_DEVICE explicit aggregate_inspector(view_t &view)
        : _inspectors(unroll_views(
              view, std::make_index_sequence<sizeof...(Inspectors)>{})) {}

    /// Inspector interface
    template <unsigned int current_id = 0, typename state_type,
              typename point3_t, typename vector3_t>
    DETRAY_HOST_DEVICE auto operator()(state_type &state,
                                       const navigation::config &cfg,
                                       const point3_t &pos,
                                       const vector3_t &dir,
                                       const char *message) {
        // Call inspector
        std::get<current_id>(_inspectors)(state, cfg, pos, dir, message);

        // Next inspector
        if constexpr (current_id <
                      std::tuple_size<inspector_tuple_t>::value - 1) {
            return operator()<current_id + 1>(state, cfg, pos, dir, message);
        }
    }

    /// @returns a specific inspector by type
    template <typename inspector_t>
    DETRAY_HOST_DEVICE constexpr decltype(auto) get() {
        return std::get<inspector_t>(_inspectors);
    }

    /// @returns a specific inspector by type - const
    template <typename inspector_t>
    DETRAY_HOST_DEVICE constexpr decltype(auto) get() const {
        return std::get<inspector_t>(_inspectors);
    }

    /// @returns a specific inspector by index
    template <std::size_t I>
    DETRAY_HOST_DEVICE constexpr decltype(auto) get() {
        return std::get<I>(_inspectors);
    }

    /// @returns a specific inspector by index - const
    template <std::size_t I>
    DETRAY_HOST_DEVICE constexpr decltype(auto) get() const {
        return std::get<I>(_inspectors);
    }

    /// @returns a tuple constructed from the inspector @param view s.
    template <concepts::device_view view_t, std::size_t... I>
    DETRAY_HOST_DEVICE constexpr auto unroll_views(
        view_t &view, std::index_sequence<I...> /*seq*/) {
        return detail::make_tuple<std::tuple>(
            Inspectors(detail::get<I>(view.m_view))...);
    }
};

namespace navigation {

namespace detail {

/// Record of a surface intersection along a track
template <typename intersetion_t>
struct candidate_record {
    using algebra_type = typename intersetion_t::algebra_type;
    using scalar_type = dscalar<algebra_type>;
    using point3_type = dpoint3D<algebra_type>;
    using vector3_type = dvector3D<algebra_type>;
    using intersection_type = intersetion_t;

    constexpr candidate_record() = default;

    /// The particle charge is not known in the navigation, but might be
    /// provided in a different context
    DETRAY_HOST_DEVICE
    constexpr candidate_record(
        const point3_type &position, const vector3_type &direction,
        const intersection_type &intr,
        const scalar_type q = detray::detail::invalid_value<scalar_type>(),
        const scalar_type p = detray::detail::invalid_value<scalar_type>())
        : pos{position},
          dir{direction},
          intersection{intr},
          charge{q},
          p_mag{p} {}

    /// Current global track position
    point3_type pos{};
    /// Current global track direction
    vector3_type dir{};
    /// The intersection result
    intersetion_t intersection{};
    /// Charge hypothesis of the particle (invalid value if not known)
    scalar_type charge{detray::detail::invalid_value<scalar_type>()};
    /// Current momentum magnitude of the particle (invalid value if not known)
    scalar_type p_mag{detray::detail::invalid_value<scalar_type>()};
};

}  // namespace detail

/// A navigation inspector that relays information about the encountered
/// objects whenever the navigator reaches one or more status flags
template <typename candidate_t, template <typename...> class vector_t = dvector,
          status... navigation_status>
struct object_tracer {

    using candidate_record_t = detail::candidate_record<candidate_t>;
    using scalar_t = typename candidate_record_t::scalar_type;

    using view_type = dvector_view<candidate_record_t>;
    using const_view_type = dvector_view<const candidate_record_t>;

    /// Default constructor
    object_tracer() = default;

    /// Device-side construction from a vecmem based view type
    DETRAY_HOST_DEVICE explicit object_tracer(
        dvector_view<candidate_record_t> &view)
        : object_trace(view) {}

    // Record all objects the navigator encounters
    vector_t<candidate_record_t> object_trace;
    dindex current_vol{dindex_invalid};
    const scalar_t inv_pos{detray::detail::invalid_value<scalar_t>()};
    typename candidate_record_t::point3_type last_pos = {inv_pos, inv_pos,
                                                         inv_pos};
    typename candidate_record_t::vector3_type last_dir = {0.f, 0.f, 0.f};

    /// Inspector interface
    template <typename state_type, typename point3_t, typename vector3_t>
    DETRAY_HOST_DEVICE auto operator()(const state_type &state,
                                       const navigation::config &,
                                       const point3_t &pos,
                                       const vector3_t &dir,
                                       const char * /*message*/) {

        // Record the candidate of an encountered object
        if ((is_status(state.status(), navigation_status) || ...)) {
            // Reached a new position: log it
            // Also log volume switches that happen without position update
            if ((getter::norm(last_pos - pos) >=
                 10.f * std::numeric_limits<scalar_t>::epsilon()) ||
                (state.is_on_portal() && current_vol != state.volume())) {

                object_trace.push_back({pos, dir, state.current()});
                last_pos = pos;
                last_dir = dir;
                current_vol = state.volume();
            }
        }
    }

    /// @returns a specific candidate from the trace
    DETRAY_HOST_DEVICE
    constexpr const candidate_record_t &operator[](std::size_t i) const {
        return object_trace[i];
    }

    /// @returns the full object trace
    DETRAY_HOST_DEVICE
    constexpr const auto &trace() const { return object_trace; }

    /// Compares a navigation status with the tracers references
    DETRAY_HOST_DEVICE
    constexpr bool is_status(const status &nav_stat, const status &ref_stat) {
        return (nav_stat == ref_stat);
    }
};

/// A navigation inspector that prints information about the current navigation
/// state. Meant for debugging.
struct print_inspector {

    using view_type = dvector_view<char>;
    using const_view_type = dvector_view<const char>;

    /// Default constructor
    print_inspector() = default;

    /// Copy constructor ensures that the string stream is set up identically
    print_inspector(const print_inspector &other)
        : debug_stream(other.debug_stream.str()) {}

    /// Move constructor
    print_inspector(print_inspector &&other) = default;

    /// Default destructor
    ~print_inspector() = default;

    /// Copy assignemten operator ensures that the string stream is set up
    /// identically
    print_inspector &operator=(const print_inspector &other) {
        // Reset
        debug_stream.str(std::string());
        debug_stream.clear();
        // Move new debug string in
        debug_stream << other.debug_stream.str();
        return *this;
    }

    /// Move assignemten operator
    print_inspector &operator=(print_inspector &&other) = default;

    /// Gathers navigation information accross navigator update calls
    std::stringstream debug_stream{};

    /// Inspector interface. Gathers detailed information during navigation
    template <typename state_type, typename point3_t, typename vector3_t>
    auto operator()(const state_type &state, const navigation::config &cfg,
                    const point3_t &track_pos, const vector3_t &track_dir,
                    const char *message) {
        std::string msg(message);
        std::string tabs = "\t\t\t\t";

        debug_stream << msg << std::endl;

        debug_stream << "Volume" << tabs << state.volume() << std::endl;
        debug_stream << "Overstep tol:\t\t\t" << cfg.overstep_tolerance
                     << std::endl;
        debug_stream << "Track pos: [r:" << getter::perp(track_pos)
                     << ", z:" << track_pos[2] << "], dir: [" << track_dir[0]
                     << ", " << track_dir[1] << ", " << track_dir[2] << "]"
                     << std::endl;
        debug_stream << "No. reachable\t\t\t" << state.n_candidates()
                     << std::endl;

        debug_stream << "Surface candidates: " << std::endl;

        using geo_ctx_t = typename state_type::detector_type::geometry_context;
        for (const auto &sf_cand : state) {

            debug_stream << sf_cand;

            // Use additional debug information that was gathered on the cand.
            if constexpr (state_type::value_type::is_debug()) {
                const auto &local = sf_cand.local;
                const auto pos =
                    tracking_surface{state.detector(), sf_cand.sf_desc}
                        .local_to_global(geo_ctx_t{}, local, track_dir);
                debug_stream << ", glob: [r:" << getter::perp(pos)
                             << ", z:" << pos[2] << "]" << std::endl;
            }
        }
        if (!state.candidates().empty()) {
            debug_stream << "=> next: ";
            if (state.is_exhausted()) {
                debug_stream << "exhausted" << std::endl;
            } else {
                debug_stream << " -> " << state.next_surface().barcode()
                             << std::endl;
            }
        }

        switch (state.status()) {
            using enum status;
            case e_abort:
                debug_stream << "status" << tabs << "abort" << std::endl;
                break;
            case e_on_target:
                debug_stream << "status" << tabs << "e_on_target" << std::endl;
                break;
            case e_unknown:
                debug_stream << "status" << tabs << "unknowm" << std::endl;
                break;
            case e_towards_object:
                debug_stream << "status" << tabs << "towards_surface"
                             << std::endl;
                break;
            case e_on_module:
                debug_stream << "status" << tabs << "on_module" << std::endl;
                break;
            case e_on_portal:
                debug_stream << "status" << tabs << "on_portal" << std::endl;
                break;
            default:
                break;
        }

        debug_stream << "current object\t\t\t";
        if (state.is_on_surface() || state.status() == status::e_on_target) {
            debug_stream << state.barcode() << std::endl;
        } else {
            debug_stream << "undefined" << std::endl;
        }

        debug_stream << "distance to next\t\t";
        if (math::fabs(state()) < static_cast<scalar>(cfg.path_tolerance)) {
            debug_stream << "on obj (within tol)" << std::endl;
        } else {
            debug_stream << state() << std::endl;
        }

        switch (state.trust_level()) {
            using enum trust_level;
            case e_no_trust:
                debug_stream << "trust" << tabs << "no_trust" << std::endl;
                break;
            case e_fair:
                debug_stream << "trust" << tabs << "fair_trust" << std::endl;
                break;
            case e_high:
                debug_stream << "trust" << tabs << "high_trust" << std::endl;
                break;
            case e_full:
                debug_stream << "trust" << tabs << "full_trust" << std::endl;
                break;
            default:
                break;
        }
        debug_stream << std::endl;
    }

    /// @returns a string representation of the gathered information
    std::string to_string() { return debug_stream.str(); }
};

}  // namespace navigation

namespace stepping {

/// A stepper inspector that prints information about the current stepper
/// state. Meant for debugging.
struct print_inspector {

    /// Default constructor
    print_inspector() = default;

    /// Copy constructor ensures that the string stream is set up identically
    print_inspector(const print_inspector &other)
        : debug_stream(other.debug_stream.str()) {}

    /// Move constructor
    print_inspector(print_inspector &&other) = default;

    /// Default destructor
    ~print_inspector() = default;

    /// Copy assignemten operator ensures that the string stream is set up
    /// identically
    print_inspector &operator=(const print_inspector &other) {
        // Reset
        debug_stream.str(std::string());
        debug_stream.clear();
        // Move new debug string in
        debug_stream << other.debug_stream.str();
        return *this;
    }

    /// Move assignemten operator
    print_inspector &operator=(print_inspector &&other) = default;

    /// Gathers stepping information from inside the stepper methods
    std::stringstream debug_stream{};

    /// Inspector interface. Gathers detailed information during stepping
    template <typename state_type>
    void operator()(const state_type &state, const stepping::config &,
                    const char *message) {
        std::string msg(message);
        std::string tabs = "\t\t\t\t";

        debug_stream << msg << std::endl;

        debug_stream << "Step size" << tabs << state.step_size() << std::endl;
        debug_stream << "Path length" << tabs << state.path_length()
                     << std::endl;

        switch (state.direction()) {
            using enum step::direction;
            case e_forward:
                debug_stream << "direction" << tabs << "forward" << std::endl;
                break;
            case e_unknown:
                debug_stream << "direction" << tabs << "unknown" << std::endl;
                break;
            case e_backward:
                debug_stream << "direction" << tabs << "backward" << std::endl;
                break;
            default:
                break;
        }

        auto pos = state().pos();

        debug_stream << "Pos:\t[r = " << math::hypot(pos[0], pos[1])
                     << ", z = " << pos[2] << "]" << std::endl;
        debug_stream << "Tangent:\t"
                     << detail::ray<ALGEBRA_PLUGIN<detray::scalar>>(state())
                     << std::endl;
        debug_stream << std::endl;
    }

    /// Inspector interface. Gathers detailed information during stepping
    template <typename state_type>
    void operator()(const state_type &state, const stepping::config &,
                    const char *message, const std::size_t n_trials,
                    const scalar step_scalor) {
        std::string msg(message);
        std::string tabs = "\t\t\t\t";

        debug_stream << msg << std::endl;

        // Remove trailing newlines
        debug_stream << "Step size" << tabs << state.step_size() << std::endl;
        debug_stream << "no. RK adjustments"
                     << "\t\t" << n_trials << std::endl;
        debug_stream << "Step size scale factor"
                     << "\t\t" << step_scalor << std::endl;

        debug_stream << std::endl;
    }

    /// @returns a string representation of the gathered information
    std::string to_string() { return debug_stream.str(); }
};

}  // namespace stepping

}  // namespace detray
