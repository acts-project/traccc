/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/core/detector.hpp"
#include "detray/definitions/detail/algorithms.hpp"
#include "detray/definitions/detail/containers.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/units.hpp"
#include "detray/geometry/barcode.hpp"
#include "detray/navigation/detail/ray.hpp"
#include "detray/navigation/intersection/intersection.hpp"
#include "detray/navigation/intersection/ray_intersector.hpp"
#include "detray/navigation/intersection_kernel.hpp"
#include "detray/navigation/navigation_config.hpp"
#include "detray/utils/ranges.hpp"

namespace detray {

namespace navigation {

/// @enum NavigationDirection
/// The navigation direction is always with
/// respect to a given momentum or direction
enum class direction : std::int_least8_t { e_backward = -1, e_forward = 1 };

/// Navigation status flags
enum class status : std::int_least8_t {
    e_abort = -3,          ///< error ocurred, propagation will be aborted
    e_on_target = -2,      ///< navigation exited successfully
    e_unknown = -1,        ///< unknown state/not initialized
    e_towards_object = 0,  ///< move towards next object
    e_on_module = 1,       ///< reached module surface
    e_on_portal = 2,       ///< reached portal surface
};

static constexpr std::size_t default_cache_size{10u};

/// A void inpector that does nothing.
///
/// Inspectors can be plugged in to understand the current navigation state.
struct void_inspector {

    struct void_view : public detail::dbase_view {};

    using view_type = void_view;
    using const_view_type = const void_view;

    constexpr void_inspector() = default;

    DETRAY_HOST_DEVICE
    constexpr explicit void_inspector(
        const void_view & /*ignored*/) { /*Do nothing*/
    }

    template <typename state_t>
    DETRAY_HOST_DEVICE constexpr void operator()(
        const state_t & /*ignored*/, const char * /*ignored*/) const {
        /*Do nothing*/
    }
};

}  // namespace navigation

/// @brief The geometry navigation class.
///
/// The navigator is initialized around a detector object, but is itself
/// agnostic to the detectors's object/primitive types.
/// Within a detector volume, the navigatior will perform a local navigation
/// based on the geometry acceleration structure(s) that are provided by the
/// volume. Once the local navigation is resolved, it moves to the next volume
/// by a portal.
/// To this end, it requires a link to the [next] navigation volume in every
/// candidate that is computed by intersection from the detector objects:
/// A module surface must link back to its mother volume, while a portal surface
/// links to the next volume in the direction of the track.
///
/// This navigator applies a trust level based update of its candidate
/// (intersection) cache, which is kept in the naviagtor's state. The trust
/// level, and with it the appropriate update policy, must be set by an actor,
/// otherwise no update will be performed.
///
/// The navigation state is set up by an init() call and then follows a
/// sequence of
/// - step()       (stepper)
/// - update()     (navigator)
/// - run_actors() (actor chain)
/// - update()     (navigator)
/// calls, which are handled by the propagator class.
///
/// The navigation heartbeat indicates, that the navigation is still running
/// and in a valid state.
///
/// @tparam detector_t the detector to navigate
/// @tparam k_cache_capacity the capacity of the candidate cache
/// @tparam inspector_t is a validation inspector that can record information
///         about the navigation state at different points of the nav. flow.
/// @tparam intersection_t candidate type
template <typename detector_t,
          std::size_t k_cache_capacity = navigation::default_cache_size,
          typename inspector_t = navigation::void_inspector,
          typename intersection_t =
              intersection2D<typename detector_t::surface_type,
                             typename detector_t::algebra_type, false>>
class navigator {

    static_assert(k_cache_capacity >= 2u,
                  "Navigation cache needs to have a capacity larger than 1");

    public:
    using detector_type = detector_t;
    using context_type = detector_type::geometry_context;

    using algebra_type = typename detector_type::algebra_type;
    using scalar_type = dscalar<algebra_type>;
    using point3_type = dpoint3D<algebra_type>;
    using vector3_type = dvector3D<algebra_type>;

    using volume_type = typename detector_type::volume_type;
    using nav_link_type = typename detector_type::surface_type::navigation_link;
    using intersection_type = intersection_t;
    using inspector_type = inspector_t;

    public:
    /// @brief A navigation state object used to cache the information of the
    /// current navigation stream.
    ///
    /// The state is passed between navigation calls and is accessible to the
    /// actors in the propagation, for which it defines the public interface
    /// towards the navigation. The navigator is responsible for updating the
    /// elements in the state's cache with every navigation call, establishing
    /// 'full trust' after changes to the track state reduced the trust level.
    class state : public detray::ranges::view_interface<state> {

        friend class navigator;

        // Allow the filling/updating of candidates
        friend struct intersection_initialize<ray_intersector>;
        friend struct intersection_update<ray_intersector>;

        using candidate_t = intersection_type;
        using candidate_cache_t = std::array<candidate_t, k_cache_capacity>;
        using candidate_itr_t = typename candidate_cache_t::iterator;
        using candidate_const_itr_t =
            typename candidate_cache_t::const_iterator;
        using dist_t = std::int_least8_t;

        public:
        using value_type = candidate_t;
        using detector_type = navigator::detector_type;

        using view_type = detail::get_view_t<inspector_t>;
        using const_view_type = detail::get_view_t<const inspector_t>;

        /// Default constructor (needs a detector)
        state() = delete;

        /// Construct using a given detector @param det
        DETRAY_HOST_DEVICE
        explicit state(const detector_type &det) : m_detector(&det) {}

        /// Constructor from detector @param det and inspector view @param view
        template <concepts::device_view view_t>
        DETRAY_HOST_DEVICE state(const detector_type &det, view_t view)
            : m_detector(&det), m_inspector(view) {}

        /// @return start position of the valid candidate range - const
        DETRAY_HOST_DEVICE
        constexpr auto begin() const -> candidate_const_itr_t {
            candidate_const_itr_t itr = m_candidates.begin();
            detray::ranges::advance(
                itr, (is_on_surface() && (m_next >= 1)) ? m_next - 1 : m_next);
            return itr;
        }

        /// @return sentinel of the valid candidate range.
        DETRAY_HOST_DEVICE
        constexpr auto end() const -> candidate_const_itr_t {
            candidate_const_itr_t itr = m_candidates.begin();
            detray::ranges::advance(itr, m_last + 1);
            return itr;
        }

        /// @returns a pointer of detector
        DETRAY_HOST_DEVICE
        const detector_type &detector() const { return (*m_detector); }

        /// @returns the navigation heartbeat
        DETRAY_HOST_DEVICE
        bool is_alive() const { return m_heartbeat; }

        /// @returns currently cached candidates - const
        DETRAY_HOST_DEVICE
        inline auto candidates() const -> const candidate_cache_t & {
            return m_candidates;
        }

        /// @returns numer of currently cached (reachable) candidates - const
        DETRAY_HOST_DEVICE
        inline auto n_candidates() const -> dindex {
            assert(m_last - m_next + 1 >= 0);
            return static_cast<dindex>(m_last - m_next + 1);
        }

        /// @returns current/previous object that was reached
        DETRAY_HOST_DEVICE
        inline auto current() const -> const candidate_t & {
            assert(m_next > 0);
            return m_candidates[static_cast<std::size_t>(m_next - 1)];
        }

        /// @returns next object that we want to reach (current target) - const
        DETRAY_HOST_DEVICE
        inline auto target() const -> const candidate_t & {
            return m_candidates[static_cast<std::size_t>(m_next)];
        }

        /// @returns last valid candidate (by position in the cache) - const
        DETRAY_HOST_DEVICE
        inline auto last() const -> const candidate_t & {
            return m_candidates[static_cast<std::size_t>(m_last)];
        }

        /// Scalar representation of the navigation state,
        /// @returns distance to next
        DETRAY_HOST_DEVICE
        scalar_type operator()() const {
            return static_cast<scalar_type>(direction()) * target().path;
        }

        /// @returns current volume (index) - const
        DETRAY_HOST_DEVICE
        inline auto volume() const -> nav_link_type { return m_volume_index; }

        /// Set start/new volume
        DETRAY_HOST_DEVICE
        inline void set_volume(dindex v) {
            assert(detail::is_invalid_value(static_cast<nav_link_type>(v)) ||
                   v < detector().volumes().size());
            if (v != m_volume_index) {
                // Make sure the new volume is properly initialized
                set_no_trust();
            }
            m_volume_index = static_cast<nav_link_type>(v);
        }

        /// @returns barcode of the detector surface the navigator is on
        /// (invalid when not on surface) - const
        DETRAY_HOST_DEVICE
        inline auto barcode() const -> geometry::barcode {
            return current().sf_desc.barcode();
        }

        /// @returns the next surface the navigator intends to reach
        DETRAY_HOST_DEVICE
        inline auto next_surface() const {
            return tracking_surface<detector_type>{*m_detector,
                                                   target().sf_desc};
        }

        /// @returns current detector surface the navigator is on
        /// (cannot be used when not on surface) - const
        DETRAY_HOST_DEVICE
        inline auto get_surface() const {
            assert(is_on_surface());
            return tracking_surface<detector_type>{*m_detector,
                                                   current().sf_desc};
        }

        /// @returns current detector volume of the navigation stream
        DETRAY_HOST_DEVICE
        inline auto get_volume() const {
            return tracking_volume<detector_type>{*m_detector, m_volume_index};
        }

        /// @returns current navigation status - const
        DETRAY_HOST_DEVICE
        inline auto status() const -> navigation::status { return m_status; }

        /// @returns current navigation direction - const
        DETRAY_HOST_DEVICE
        inline auto direction() const -> navigation::direction {
            return m_direction;
        }

        /// Set direction
        DETRAY_HOST_DEVICE
        inline void set_direction(const navigation::direction dir) {
            m_direction = dir;
        }

        /// @returns navigation trust level - const
        DETRAY_HOST_DEVICE
        inline auto trust_level() const -> navigation::trust_level {
            return m_trust_level;
        }

        /// Update navigation trust level to no trust
        DETRAY_HOST_DEVICE
        inline void set_no_trust() {
            m_trust_level = navigation::trust_level::e_no_trust;
        }

        /// Update navigation trust level to full trust
        DETRAY_HOST_DEVICE
        inline void set_full_trust() {
            m_trust_level = m_trust_level <= navigation::trust_level::e_full
                                ? m_trust_level
                                : navigation::trust_level::e_full;
        }

        /// Update navigation trust level to high trust
        DETRAY_HOST_DEVICE
        inline void set_high_trust() {
            m_trust_level = m_trust_level <= navigation::trust_level::e_high
                                ? m_trust_level
                                : navigation::trust_level::e_high;
        }

        /// Update navigation trust level to fair trust
        DETRAY_HOST_DEVICE
        inline void set_fair_trust() {
            m_trust_level = m_trust_level <= navigation::trust_level::e_fair
                                ? m_trust_level
                                : navigation::trust_level::e_fair;
        }

        /// Helper method to check the track has reached a module surface
        DETRAY_HOST_DEVICE
        inline auto is_on_surface() const -> bool {
            return (m_status == navigation::status::e_on_module ||
                    m_status == navigation::status::e_on_portal);
        }

        /// Helper method to check the track has reached a sensitive surface
        DETRAY_HOST_DEVICE
        inline auto is_on_sensitive() const -> bool {
            return (m_status == navigation::status::e_on_module) &&
                   (barcode().id() == surface_id::e_sensitive);
        }

        /// Helper method to check the track has reached a passive surface
        DETRAY_HOST_DEVICE
        inline auto is_on_passive() const -> bool {
            return (m_status == navigation::status::e_on_module) &&
                   (barcode().id() == surface_id::e_passive);
        }

        /// Helper method to check the track has reached a portal surface
        DETRAY_HOST_DEVICE
        inline auto is_on_portal() const -> bool {
            return m_status == navigation::status::e_on_portal;
        }

        /// Helper method to check the track has encountered material
        DETRAY_HOST_DEVICE
        inline auto encountered_sf_material() const -> bool {
            return (is_on_surface()) && (current().sf_desc.material().id() !=
                                         detector_t::materials::id::e_none);
        }

        /// Helper method to check if a kernel is exhausted - const
        DETRAY_HOST_DEVICE
        inline auto is_exhausted() const -> bool {
            return n_candidates() == 0u;
        }

        /// @returns flag that indicates whether navigation was successful
        DETRAY_HOST_DEVICE
        inline auto is_complete() const -> bool {
            // Normal exit for this navigation?
            return m_status == navigation::status::e_on_target && !m_heartbeat;
        }

        /// @returns the navigation inspector - const
        DETRAY_HOST_DEVICE
        inline const auto &inspector() const { return m_inspector; }

        /// @returns the navigation inspector
        DETRAY_HOST_DEVICE
        inline auto &inspector() { return m_inspector; }

        /// Navigation state that cannot be recovered from. Leave the other
        /// data for inspection.
        ///
        /// @return navigation heartbeat (dead)
        DETRAY_HOST_DEVICE
        inline auto abort() -> bool {
            m_status = navigation::status::e_abort;
            m_heartbeat = false;
            // Don't do anything if aborted
            m_trust_level = navigation::trust_level::e_full;
            run_inspector({}, point3_type{0.f, 0.f, 0.f},
                          vector3_type{0.f, 0.f, 0.f}, "Aborted: ");
            return m_heartbeat;
        }

        /// Navigation reaches final target or leaves detector world. Stop
        /// navigation.
        ///
        /// @return navigation heartbeat (dead)
        DETRAY_HOST_DEVICE
        inline auto exit() -> bool {
            m_status = navigation::status::e_on_target;
            m_heartbeat = false;
            m_trust_level = navigation::trust_level::e_full;
            run_inspector({}, point3_type{0.f, 0.f, 0.f},
                          vector3_type{0.f, 0.f, 0.f}, "Exited: ");
            this->clear();
            return m_heartbeat;
        }

        private:
        /// @return start position of valid candidate range.
        DETRAY_HOST_DEVICE
        constexpr auto begin() -> candidate_itr_t {
            candidate_itr_t itr = m_candidates.begin();
            detray::ranges::advance(
                itr, (is_on_surface() && (m_next >= 1)) ? m_next - 1 : m_next);
            return itr;
        }

        /// @return sentinel of the valid candidate range.
        DETRAY_HOST_DEVICE
        constexpr auto end() -> candidate_itr_t {
            candidate_itr_t itr = m_candidates.begin();
            detray::ranges::advance(itr, m_last + 1);
            return itr;
        }

        /// Helper method to check if a candidate lies on a surface - const
        DETRAY_HOST_DEVICE inline auto is_on_surface(
            const intersection_type &candidate,
            const navigation::config &cfg) const -> bool {
            return (math::fabs(candidate.path) < cfg.path_tolerance);
        }

        /// @returns currently cached candidates
        DETRAY_HOST_DEVICE
        inline auto candidates() -> candidate_cache_t & { return m_candidates; }

        /// Insert a new element @param new_cadidate before position @param pos
        DETRAY_HOST_DEVICE
        constexpr void insert(candidate_itr_t pos,
                              const intersection_type &new_cadidate) {

            // Candidate is too far away to be placed in cache
            if (pos == m_candidates.end()) {
                return;
            }

            assert(detail::is_invalid_value(new_cadidate.volume_link) ||
                   new_cadidate.volume_link < detector().volumes().size());

            // Insert the first candidate
            if (n_candidates() == 0) {
                m_candidates[0] = new_cadidate;
                ++m_last;
                assert(m_next <= m_last + 1);
                assert(static_cast<std::size_t>(m_last) < k_cache_capacity);
                return;
            }

            // Position where to insert the new candidate
            auto idx{static_cast<dist_t>(
                detray::ranges::distance(m_candidates.begin(), pos))};
            assert(idx >= 0);

            // Shift all following candidates and evict the last element,
            // if the cache is already full
            constexpr auto shift_max{static_cast<dist_t>(k_cache_capacity - 2)};
            const dist_t shift_begin{math::min(m_last, shift_max)};

            for (dist_t i = shift_begin; i >= idx; --i) {
                const auto j{static_cast<std::size_t>(i)};
                m_candidates[j + 1u] = m_candidates[j];
            }

            // Now insert the new candidate and update candidate range
            m_candidates[static_cast<std::size_t>(idx)] = new_cadidate;
            m_last = math::min(static_cast<dist_t>(m_last + 1),
                               static_cast<dist_t>(k_cache_capacity - 1));

            assert(m_next <= m_last + 1);
            assert(static_cast<std::size_t>(m_last) < k_cache_capacity);
        }

        /// @returns next object that we want to reach (current target)
        DETRAY_HOST_DEVICE
        inline auto target() -> candidate_t & {
            return m_candidates[static_cast<std::size_t>(m_next)];
        }

        /// @returns last valid candidate (by position in the cache)
        DETRAY_HOST_DEVICE
        inline auto last() -> candidate_t & {
            return m_candidates[static_cast<std::size_t>(m_last)];
        }

        /// Set the next surface that we want to reach (update target)
        DETRAY_HOST_DEVICE
        inline auto next() -> void {
            ++m_next;
            assert(m_next <= m_last + 1);
            assert(m_next < static_cast<dist_t>(k_cache_capacity) + 1);
        }

        /// Set the next surface that we want to reach (update target)
        DETRAY_HOST_DEVICE
        inline void set_next(dindex pos) {
            m_next = pos;
            assert(m_next <= m_last + 1);
            assert(m_next < static_cast<dist_t>(k_cache_capacity) + 1);
        }

        /// Set the next surface that we want to reach (update target)
        DETRAY_HOST_DEVICE
        inline void set_next(candidate_itr_t new_next) {
            m_next = static_cast<dist_t>(
                detray::ranges::distance(m_candidates.begin(), new_next));
            assert(m_next < static_cast<dist_t>(k_cache_capacity));
        }

        /// Updates the position of the last valid candidate
        DETRAY_HOST_DEVICE
        inline void set_last(candidate_itr_t new_last) {
            m_last = static_cast<dist_t>(
                detray::ranges::distance(m_candidates.begin(), new_last) - 1);
            assert(m_next <= m_last + 1);
            assert(m_last < static_cast<dist_t>(k_cache_capacity));
        }

        /// Clear the state
        DETRAY_HOST_DEVICE
        inline void clear() {
            // Mark all data in the cache as unreachable
            for (std::size_t i = 0u; i < k_cache_capacity; ++i) {
                m_candidates[i].path = std::numeric_limits<scalar_type>::max();
            }
            m_next = 0;
            m_last = -1;
        }

        /// Call the navigation inspector
        DETRAY_HOST_DEVICE
        inline void run_inspector(
            [[maybe_unused]] const navigation::config &cfg,
            [[maybe_unused]] const point3_type &track_pos,
            [[maybe_unused]] const vector3_type &track_dir,
            [[maybe_unused]] const char *message) {
            if constexpr (!std::is_same_v<inspector_t,
                                          navigation::void_inspector>) {
                m_inspector(*this, cfg, track_pos, track_dir, message);
            }
        }

        /// Our cache of candidates (intersections with any kind of surface)
        candidate_cache_t m_candidates;

        /// Detector pointer
        const detector_type *m_detector{nullptr};

        /// Index in the detector volume container of current navigation volume
        nav_link_type m_volume_index{0u};

        /// The next best candidate (target): m_next <= m_last + 1.
        /// m_next can be larger than m_last when the cache is exhausted
        dist_t m_next{0};

        /// The last reachable candidate: m_last < k_cache_capacity
        /// Can never be advanced beyond the last element
        dist_t m_last{-1};

        /// The navigation status
        navigation::status m_status{navigation::status::e_unknown};

        /// The navigation trust level determines how this states cache is to
        /// be updated in the current navigation call
        navigation::trust_level m_trust_level{
            navigation::trust_level::e_no_trust};

        /// The navigation direction
        navigation::direction m_direction{navigation::direction::e_forward};

        /// Heartbeat of this navigation flow signals navigation is alive
        bool m_heartbeat{false};

        /// The inspector type of this navigation engine
        [[no_unique_address]] inspector_type m_inspector;
    };

    private:
    /// A functor that fills the navigation candidates vector by intersecting
    /// the surfaces in the volume neighborhood
    struct candidate_search {

        /// Test the volume links
        template <typename track_t>
        DETRAY_HOST_DEVICE void operator()(
            const typename detector_type::surface_type &sf_descr,
            const detector_type &det, const context_type &ctx,
            const track_t &track, state &nav_state,
            const std::array<scalar_type, 2> mask_tol,
            const scalar_type mask_tol_scalor,
            const scalar_type overstep_tol) const {

            const auto sf = tracking_surface{det, sf_descr};

            sf.template visit_mask<intersection_initialize<ray_intersector>>(
                nav_state,
                detail::ray<algebra_type>(
                    track.pos(),
                    static_cast<scalar_type>(nav_state.direction()) *
                        track.dir()),
                sf_descr, det.transform_store(), ctx,
                sf.is_portal() ? std::array<scalar_type, 2>{0.f, 0.f}
                               : mask_tol,
                mask_tol_scalor, overstep_tol);
        }
    };

    public:
    /// @brief Helper method to initialize a volume.
    ///
    /// Calls the volumes accelerator structure for local navigation, then tests
    /// the surfaces for intersection and sorts the reachable candidates to find
    /// the clostest one (next candidate).
    ///
    /// @tparam track_t type of track, needs to provide pos() and dir() methods
    ///
    /// @param track access to the track parameters
    /// @param state the current navigation state
    /// @param cfg the navigation configuration
    template <typename track_t>
    DETRAY_HOST_DEVICE inline void init(const track_t &track, state &navigation,
                                        const navigation::config &cfg,
                                        const context_type &ctx) const {
        const auto &det = navigation.detector();
        const auto volume = tracking_volume{det, navigation.volume()};

        // Clean up state
        navigation.clear();
        navigation.m_heartbeat = true;

        // Search for neighboring surfaces and fill candidates into cache
        volume.template visit_neighborhood<candidate_search>(
            track, cfg, ctx, det, ctx, track, navigation,
            std::array<scalar_type, 2u>{cfg.min_mask_tolerance,
                                        cfg.max_mask_tolerance},
            static_cast<scalar_type>(cfg.mask_tolerance_scalor),
            static_cast<scalar_type>(cfg.overstep_tolerance));

        // Determine overall state of the navigation after updating the cache
        update_navigation_state(navigation, cfg);

        // If init was not successful, the propagation setup is broken
        if (navigation.trust_level() != navigation::trust_level::e_full) {
            navigation.m_heartbeat = false;
        }

        navigation.run_inspector(cfg, track.pos(), track.dir(),
                                 "Init complete: ");
    }

    /// @brief Complete update of the navigation flow.
    ///
    /// Restores 'full trust' state to the cadidates cache and checks whether
    /// the track stepped onto a portal and a volume switch is due. If so, or
    /// when the previous update according to the given trust level
    /// failed to restore trust, it performs a complete reinitialization of the
    /// navigation.
    ///
    /// @tparam track_t type of track, needs to provide pos() and dir() methods
    ///
    /// @param track access to the track parameters
    /// @param state the current navigation state
    /// @param cfg the navigation configuration
    ///
    /// @returns a heartbeat to indicate if the navigation is still alive
    template <typename track_t>
    DETRAY_HOST_DEVICE inline bool update(const track_t &track,
                                          state &navigation,
                                          const navigation::config &cfg,
                                          const context_type &ctx = {}) const {
        // Candidates are re-evaluated based on the current trust level.
        // Should result in 'full trust'
        bool is_init = update_kernel(track, navigation, cfg, ctx);

        // Update was completely successful (most likely case)
        if (navigation.trust_level() == navigation::trust_level::e_full) {
            return is_init;
        }
        // Otherwise: did we run into a portal?
        else if (navigation.is_on_portal()) {
            // Set volume index to the next volume provided by the portal
            navigation.set_volume(navigation.current().volume_link);

            // Navigation reached the end of the detector world
            if (detail::is_invalid_value(navigation.volume())) {
                navigation.exit();
                return is_init;
            }

            // Either end of world or valid volume index
            assert(detail::is_invalid_value(navigation.volume()) ||
                   navigation.volume() <
                       navigation.detector().volumes().size());

            // Run inspection when needed (keep for debugging)
            // navigation.run_inspector(cfg, track.pos(), track.dir(), "Volume
            // switch: ");

            init(track, navigation, cfg, ctx);
            is_init = true;

            // Fresh initialization, reset trust and hearbeat even though we are
            // on inner portal
            navigation.m_trust_level = navigation::trust_level::e_full;
            navigation.m_heartbeat = !navigation.is_exhausted();
        }
        // If no trust could be restored for the current state, (local)
        // navigation might be exhausted: re-initialize volume
        else {
            init(track, navigation, cfg, ctx);
            is_init = true;

            // Sanity check: Should never be the case after complete update call
            if (navigation.trust_level() != navigation::trust_level::e_full) {
                // Try to save the navigation flow: Look further behind the
                // track
                auto loose_cfg{cfg};
                // Use the max mask tolerance in case a track leaves the volume
                // when a sf is 'sticking' out of the portals due to the tol
                loose_cfg.overstep_tolerance =
                    math::min(100.f * cfg.overstep_tolerance,
                              -10.f * cfg.max_mask_tolerance);

                init(track, navigation, loose_cfg, ctx);

                // Unrecoverable
                if (navigation.trust_level() !=
                    navigation::trust_level::e_full) {
                    navigation.abort();
                }
            }
        }

        return is_init;
    }

    private:
    /// Helper method to update the candidates (surface intersections)
    /// based on an externally provided trust level. Will (re-)initialize the
    /// navigation if there is no trust.
    ///
    /// @tparam track_t type of track, needs to provide pos() and dir() methods
    ///
    /// @param track access to the track parameters
    /// @param state the current navigation state
    /// @param cfg the navigation configuration
    template <typename track_t>
    DETRAY_HOST_DEVICE inline bool update_kernel(
        const track_t &track, state &navigation, const navigation::config &cfg,
        const context_type &ctx) const {

        const auto &det = navigation.detector();

        // Current candidates are up to date, nothing left to do
        if (navigation.trust_level() == navigation::trust_level::e_full) {
            return false;
        }

        // Update only the current candidate and the corresponding next target
        // - do this only when the navigation state is still coherent
        if (navigation.trust_level() == navigation::trust_level::e_high) {
            // Update next candidate: If not reachable, 'high trust' is broken
            if (!update_candidate(navigation.direction(), navigation.target(),
                                  track, det, cfg, ctx)) {
                navigation.m_status = navigation::status::e_unknown;
                navigation.set_fair_trust();
            } else {

                // Update navigation flow on the new candidate information
                update_navigation_state(navigation, cfg);

                navigation.run_inspector(cfg, track.pos(), track.dir(),
                                         "Update complete: high trust: ");

                // The work is done if: the track has not reached a surface yet
                // or trust is gone (portal was reached or the cache is broken).
                if (navigation.status() ==
                        navigation::status::e_towards_object ||
                    navigation.trust_level() ==
                        navigation::trust_level::e_no_trust) {
                    return false;
                }

                // Else: Track is on module.
                // Ready the next candidate after the current module
                if (update_candidate(navigation.direction(),
                                     navigation.target(), track, det, cfg,
                                     ctx)) {
                    return false;
                }

                // If next candidate is not reachable, don't 'return', but
                // escalate the trust level.
                // This will run into the fair trust case below.
                navigation.set_fair_trust();
            }
        }

        // Re-evaluate all currently available candidates and sort again
        // - do this when your navigation state is stale, but not invalid
        if (navigation.trust_level() == navigation::trust_level::e_fair) {

            for (auto &candidate : navigation) {
                // Disregard this candidate if it is not reachable
                if (!update_candidate(navigation.direction(), candidate, track,
                                      det, cfg, ctx)) {
                    // Forcefully set dist to numeric max for sorting
                    candidate.path = std::numeric_limits<scalar_type>::max();
                }
            }
            detail::sequential_sort(navigation.begin(), navigation.end());
            // Take the nearest (sorted) candidate first
            navigation.set_next(navigation.begin());
            // Ignore unreachable elements (needed to determine exhaustion)
            navigation.set_last(find_invalid(navigation.candidates()));
            // Update navigation flow on the new candidate information
            update_navigation_state(navigation, cfg);

            navigation.run_inspector(cfg, track.pos(), track.dir(),
                                     "Update complete: fair trust: ");

            if (!navigation.is_exhausted()) {
                return false;
            }
        }

        // Actor flagged cache as broken (other cases of 'no trust' are
        // handeled after volume switch was checked in 'update()')
        if (navigation.trust_level() == navigation::trust_level::e_no_trust) {
            init(track, navigation, cfg, ctx);
            return true;
        }

        return false;
    }

    /// @brief Helper method that re-establishes the navigation state after an
    /// update.
    ///
    /// It checks wether the track has reached a surface or is still moving
    /// towards the next surface candidate. If no new next candidate can be
    //  found, it flags 'no trust' in order to trigger a volume initialization.
    ///
    /// @param state the current navigation state
    /// @param cfg the navigation configuration
    DETRAY_HOST_DEVICE inline void update_navigation_state(
        state &navigation, const navigation::config &cfg) const {

        // Check whether the track reached the current candidate. Might be a
        // portal, in which case the navigation needs to be re-initialized
        if (!navigation.is_exhausted() &&
            navigation.is_on_surface(navigation.target(), cfg)) {
            // Set the next object that we want to reach (this function is only
            // called once the cache has been updated to a full trust state).
            // Might lead to exhausted cache.
            navigation.next();
            navigation.m_status = (navigation.current().sf_desc.is_portal())
                                      ? navigation::status::e_on_portal
                                      : navigation::status::e_on_module;
        } else {
            // Otherwise the track is moving towards a surface
            navigation.m_status = navigation::status::e_towards_object;
        }
        // Exhaustion happens when after an update no next candidate in the
        // cache is reachable anymore -> triggers init of [new] volume
        // In backwards navigation or with strongly bent tracks, the cache may
        // not be exhausted when trying to exit the volume (the ray is seeing
        // the opposite side of the volume)
        navigation.m_trust_level =
            navigation.is_exhausted() || navigation.is_on_portal()
                ? navigation::trust_level::e_no_trust
                : navigation::trust_level::e_full;
    }

    /// @brief Helper method that updates the intersection of a single candidate
    /// and checks reachability
    ///
    /// @tparam track_t type of track, needs to provide pos() and dir() methods
    ///
    /// @param candidate the candidate intersection to be updated
    /// @param track access to the track parameters
    /// @param det access to the detector (geometry)
    /// @param cfg the navigation configuration
    ///
    /// @returns whether the track can reach this candidate.
    template <typename track_t>
    DETRAY_HOST_DEVICE inline bool update_candidate(
        const navigation::direction nav_dir, intersection_type &candidate,
        const track_t &track, const detector_type &det,
        const navigation::config &cfg, const context_type &ctx) const {

        if (candidate.sf_desc.barcode().is_invalid()) {
            return false;
        }

        const auto sf = tracking_surface{det, candidate.sf_desc};

        // Check whether this candidate is reachable by the track
        return sf.template visit_mask<intersection_update<ray_intersector>>(
            detail::ray<algebra_type>(
                track.pos(), static_cast<scalar_type>(nav_dir) * track.dir()),
            candidate, det.transform_store(), ctx,
            sf.is_portal() ? std::array<scalar_type, 2>{0.f, 0.f}
                           : std::array<scalar_type, 2>{cfg.min_mask_tolerance,
                                                        cfg.max_mask_tolerance},
            static_cast<scalar_type>(cfg.mask_tolerance_scalor),
            static_cast<scalar_type>(cfg.overstep_tolerance));
    }

    /// Helper to evict all unreachable/invalid candidates from the cache:
    /// Finds the first unreachable candidate (has been invalidated during
    /// update) in a sorted (!) cache.
    ///
    /// @param candidates the cache of candidates to be cleaned
    DETRAY_HOST_DEVICE inline auto find_invalid(
        typename state::candidate_cache_t &candidates) const {
        // Depends on previous invalidation of unreachable candidates!
        auto not_reachable = [](const intersection_type &candidate) {
            return candidate.path == std::numeric_limits<scalar_type>::max();
        };

        return detail::find_if(candidates.begin(), candidates.end(),
                               not_reachable);
    }
};

}  // namespace detray
