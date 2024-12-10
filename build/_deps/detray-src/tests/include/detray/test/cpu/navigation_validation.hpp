/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/detectors/bfield.hpp"
#include "detray/navigation/detail/ray.hpp"
#include "detray/propagator/line_stepper.hpp"
#include "detray/propagator/rk_stepper.hpp"
#include "detray/tracks/tracks.hpp"

// Detray test include(s)
#include "detray/test/common/fixture_base.hpp"
#include "detray/test/common/navigation_validation_config.hpp"
#include "detray/test/validation/detector_scan_utils.hpp"
#include "detray/test/validation/material_validation_utils.hpp"
#include "detray/test/validation/navigation_validation_utils.hpp"

// System include(s)
#include <iostream>
#include <string>

namespace detray::test {

/// @brief Test class that runs the navigation check on a given detector.
///
/// @note The lifetime of the detector needs to be guaranteed.
template <typename detector_t, template <typename> class scan_type>
class navigation_validation : public test::fixture_base<> {

    using scalar_t = typename detector_t::scalar_type;
    using algebra_t = typename detector_t::algebra_type;
    using vector3_t = typename detector_t::vector3_type;
    using free_track_parameters_t = free_track_parameters<algebra_t>;
    using trajectory_type = typename scan_type<algebra_t>::trajectory_type;
    using truth_trace_t = typename scan_type<
        algebra_t>::template intersection_trace_type<detector_t>;

    /// Switch between rays and helices
    static constexpr auto k_use_rays{
        std::is_same_v<detail::ray<algebra_t>, trajectory_type>};

    public:
    using fixture_type = test::fixture_base<>;
    using config = navigation_validation_config;

    explicit navigation_validation(
        const detector_t &det, const typename detector_t::name_map &names,
        const config &cfg = {},
        const typename detector_t::geometry_context gctx = {})
        : m_cfg{cfg}, m_gctx{gctx}, m_det{det}, m_names{names} {

        if (!m_cfg.whiteboard()) {
            throw std::invalid_argument("No white board was passed to " +
                                        m_cfg.name() + " test");
        }
    }

    /// Run the check
    void TestBody() override {
        using namespace detray;
        using namespace navigation;

        using intersection_t =
            typename truth_trace_t::value_type::intersection_type;

        // Runge-Kutta stepper
        using hom_bfield_t = bfield::const_field_t;
        using bfield_t =
            std::conditional_t<k_use_rays, navigation_validator::empty_bfield,
                               hom_bfield_t>;
        using rk_stepper_t =
            rk_stepper<typename hom_bfield_t::view_t, algebra_t,
                       unconstrained_step, stepper_rk_policy,
                       stepping::print_inspector>;
        using line_stepper_t =
            line_stepper<algebra_t, unconstrained_step, stepper_default_policy,
                         stepping::print_inspector>;
        using stepper_t =
            std::conditional_t<k_use_rays, line_stepper_t, rk_stepper_t>;

        bfield_t b_field{};
        if constexpr (!k_use_rays) {
            b_field = bfield::create_const_field(m_cfg.B_vector());
        }

        // Use ray or helix
        const std::string det_name{m_det.name(m_names)};
        const std::string truth_data_name{
            k_use_rays ? det_name + "_ray_scan" : det_name + "_helix_scan"};

        /// Collect some statistics
        std::size_t n_tracks{0u}, n_surfaces{0u}, n_miss_nav{0u},
            n_miss_truth{0u}, n_matching_error{0u}, n_fatal{0u};

        std::cout << "\nINFO: Fetching data from white board..." << std::endl;
        if (!m_cfg.whiteboard()->exists(truth_data_name)) {
            throw std::runtime_error(
                "White board is empty! Please run detector scan first");
        }
        auto &truth_traces =
            m_cfg.whiteboard()->template get<std::vector<truth_trace_t>>(
                truth_data_name);

        std::size_t n_test_tracks{
            std::min(m_cfg.n_tracks(), truth_traces.size())};
        std::cout << "\nINFO: Running navigation validation on: " << det_name
                  << "...\n"
                  << std::endl;

        const std::string prefix{k_use_rays ? det_name + "_ray_"
                                            : det_name + "_helix_"};
        std::ios_base::openmode io_mode = std::ios::trunc | std::ios::out;
        const std::string debug_file_name{prefix + "navigation_validation.txt"};
        detray::io::file_handle debug_file{debug_file_name, io_mode};

        // Keep a record of track positions and material along the track
        dvector<dvector<navigation::detail::candidate_record<intersection_t>>>
            recorded_traces{};
        dvector<material_validator::material_record<scalar_t>> mat_records{};
        std::vector<std::pair<trajectory_type, std::vector<intersection_t>>>
            missed_intersections{};

        scalar_t min_pT{std::numeric_limits<scalar_t>::max()};
        scalar_t max_pT{-std::numeric_limits<scalar_t>::max()};
        for (auto &truth_trace : truth_traces) {

            if (n_tracks >= m_cfg.n_tracks()) {
                break;
            }

            // Follow the test trajectory with a track and check, if we find
            // the same volumes and distances along the way
            const auto &start = truth_trace.front();
            const auto &track = start.track_param;
            trajectory_type test_traj = get_parametrized_trajectory(track);

            const scalar q = start.charge;
            // If the momentum is unknown, 1 GeV is the safest option to keep
            // the direction vector normalized
            const scalar pT{q == 0.f ? 1.f * unit<scalar>::GeV : track.pT(q)};
            const scalar p{q == 0.f ? 1.f * unit<scalar>::GeV : track.p(q)};
            min_pT = std::min(min_pT, pT);
            max_pT = std::max(max_pT, pT);

            // Run the propagation
            auto [success, obj_tracer, step_trace, mat_record, mat_trace,
                  nav_printer, step_printer] =
                navigation_validator::record_propagation<stepper_t>(
                    m_gctx, &m_host_mr, m_det, m_cfg.propagation(), track,
                    b_field);

            if (success) {
                // The navigator does not record the initial track position:
                // add it as a dummy record
                obj_tracer.object_trace.insert(
                    obj_tracer.object_trace.begin(),
                    {track.pos(), track.dir(), start.intersection});

                // Adjust the track charge, which is unknown to the navigation
                for (auto &record : obj_tracer.object_trace) {
                    record.charge = q;
                    record.p_mag = p;
                }

                auto [result, n_missed_nav, n_missed_truth, n_error,
                      missed_inters] =
                    navigation_validator::compare_traces(
                        truth_trace, obj_tracer.object_trace, test_traj,
                        n_tracks, n_test_tracks, &(*debug_file));

                missed_intersections.push_back(
                    std::make_pair(test_traj, std::move(missed_inters)));

                // Update statistics
                success &= result;
                n_miss_nav += n_missed_nav;
                n_miss_truth += n_missed_truth;
                n_matching_error += n_error;

            } else {
                // Propagation did not succeed
                ++n_fatal;

                std::vector<intersection_t> missed_inters{};
                missed_intersections.push_back(
                    std::make_pair(test_traj, missed_inters));
            }

            if (!success) {
                // Write debug info to file
                *debug_file << "TEST TRACK " << n_tracks << ":\n\n"
                            << nav_printer.to_string()
                            << step_printer.to_string();

                detector_scanner::display_error(
                    m_gctx, m_det, m_names, m_cfg.name(), test_traj,
                    truth_trace, m_cfg.svg_style(), n_tracks, n_test_tracks,
                    obj_tracer.object_trace);
            }

            recorded_traces.push_back(std::move(obj_tracer.object_trace));
            mat_records.push_back(mat_record);

            EXPECT_TRUE(success) << "INFO: Wrote navigation debugging data in: "
                                 << debug_file_name;

            ++n_tracks;

            ASSERT_EQ(truth_trace.size(), recorded_traces.back().size());
            n_surfaces += truth_trace.size();
        }

        // Calculate and display the result
        navigation_validator::print_efficiency(n_tracks, n_surfaces, n_miss_nav,
                                               n_miss_truth, n_fatal,
                                               n_matching_error);

        // Print track positions for plotting
        std::string mometum_str{std::to_string(min_pT) + "_" +
                                std::to_string(max_pT)};

        const auto data_path{
            std::filesystem::path{m_cfg.track_param_file()}.parent_path()};
        const auto truth_trk_path{data_path / (prefix + "truth_track_params_" +
                                               mometum_str + "GeV.csv")};
        const auto trk_path{data_path / (prefix + "navigation_track_params_" +
                                         mometum_str + "GeV.csv")};
        const auto mat_path{data_path / (prefix + "accumulated_material_" +
                                         mometum_str + "GeV.csv")};
        const auto missed_path{
            data_path /
            (prefix + "missed_intersections_dists_" + mometum_str + "GeV.csv")};

        // Write the distance of the missed intersection local position
        // to the surface boundaries to file for plotting
        navigation_validator::write_dist_to_boundary(
            m_det, m_names, missed_path.string(), missed_intersections);
        detector_scanner::write_tracks(truth_trk_path.string(), truth_traces);
        navigation_validator::write_tracks(trk_path.string(), recorded_traces);
        material_validator::write_material(mat_path.string(), mat_records);

        std::cout
            << "INFO: Wrote distance to boundary of missed intersections to: "
            << missed_path << std::endl;
        std::cout << "INFO: Wrote truth track states in: " << truth_trk_path
                  << std::endl;
        std::cout << "INFO: Wrote recorded track states in: " << trk_path
                  << std::endl;
        std::cout << "INFO: Wrote accumulated material in: " << mat_path
                  << std::endl;
    }

    private:
    /// @returns either the helix or ray corresponding to the input track
    /// parameters @param track
    trajectory_type get_parametrized_trajectory(
        const free_track_parameters_t &track) {
        std::unique_ptr<trajectory_type> test_traj{nullptr};
        if constexpr (k_use_rays) {
            test_traj = std::make_unique<trajectory_type>(track);
        } else {
            test_traj =
                std::make_unique<trajectory_type>(track, &(m_cfg.B_vector()));
        }
        return *(test_traj.release());
    }

    /// The configuration of this test
    config m_cfg;
    /// The geometry context to check
    typename detector_t::geometry_context m_gctx{};
    /// Vecmem memory resource for the host allocations
    vecmem::host_memory_resource m_host_mr{};
    /// The detector to be checked
    const detector_t &m_det;
    /// Volume names
    const typename detector_t::name_map &m_names;
};

template <typename detector_t>
using straight_line_navigation =
    navigation_validation<detector_t, detray::ray_scan>;

template <typename detector_t>
using helix_navigation = navigation_validation<detector_t, detray::helix_scan>;

}  // namespace detray::test
