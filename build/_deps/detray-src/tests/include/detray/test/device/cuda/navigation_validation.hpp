/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/navigation/detail/ray.hpp"
#include "detray/propagator/line_stepper.hpp"
#include "detray/propagator/rk_stepper.hpp"
#include "detray/tracks/tracks.hpp"

// Detray test include(s)
#include "detray/test/common/fixture_base.hpp"
#include "detray/test/common/navigation_validation_config.hpp"
#include "detray/test/utils/inspectors.hpp"
#include "detray/test/validation/detector_scan_utils.hpp"
#include "detray/test/validation/detector_scanner.hpp"
#include "detray/test/validation/material_validation_utils.hpp"
#include "detray/test/validation/navigation_validation_utils.hpp"

// Vecmem include(s)
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// System include(s)
#include <tuple>

namespace detray::cuda {

/// Launch the navigation validation kernel
///
/// @param[in] det_view the detector vecmem view
/// @param[in] cfg the propagation configuration
/// @param[in] field_data the magentic field view (maybe an empty field)
/// @param[in] truth_intersection_traces_view vecemem view of the truth data
/// @param[out] recorded_intersections_view vecemem view of the intersections
///                                         recorded by the navigator
template <typename bfield_t, typename detector_t,
          typename intersection_record_t>
void navigation_validation_device(
    typename detector_t::view_type det_view, const propagation::config &cfg,
    bfield_t field_data,
    vecmem::data::jagged_vector_view<const intersection_record_t>
        &truth_intersection_traces_view,
    vecmem::data::jagged_vector_view<navigation::detail::candidate_record<
        typename intersection_record_t::intersection_type>>
        &recorded_intersections_view,
    vecmem::data::vector_view<
        material_validator::material_record<typename detector_t::scalar_type>>
        &mat_records_view,
    vecmem::data::jagged_vector_view<
        material_validator::material_params<typename detector_t::scalar_type>>
        &mat_steps_view);

/// Prepare data for device navigation run
template <typename bfield_t, typename detector_t,
          typename intersection_record_t>
inline auto run_navigation_validation(
    vecmem::memory_resource *host_mr, vecmem::memory_resource *dev_mr,
    const detector_t &det, const propagation::config &cfg, bfield_t field_data,
    const std::vector<std::vector<intersection_record_t>>
        &truth_intersection_traces) {

    using intersection_t = typename intersection_record_t::intersection_type;
    using scalar_t = typename detector_t::scalar_type;
    using material_record_t = material_validator::material_record<scalar_t>;
    using material_params_t = material_validator::material_params<scalar_t>;

    // Helper object for performing memory copies (to CUDA devices)
    vecmem::cuda::copy cuda_cpy;

    // Copy the detector to device and get its view
    auto det_buffer = detray::get_buffer(det, *dev_mr, cuda_cpy);
    auto det_view = detray::get_data(det_buffer);

    // Move truth intersection traces data to device
    auto truth_intersection_traces_data =
        vecmem::get_data(truth_intersection_traces, host_mr);
    auto truth_intersection_traces_buffer =
        cuda_cpy.to(truth_intersection_traces_data, *dev_mr, host_mr,
                    vecmem::copy::type::host_to_device);
    vecmem::data::jagged_vector_view<const intersection_record_t>
        truth_intersection_traces_view =
            vecmem::get_data(truth_intersection_traces_buffer);

    // Buffer for the intersections recorded by the navigator
    std::vector<std::size_t> capacities;
    for (const auto &trace : truth_intersection_traces) {
        // Increase the capacity, in case the navigator finds more surfaces
        // than the truth intersections (usually just one)
        capacities.push_back(trace.size() + 10u);
    }

    vecmem::data::jagged_vector_buffer<
        navigation::detail::candidate_record<intersection_t>>
        recorded_intersections_buffer(capacities, *dev_mr, host_mr,
                                      vecmem::data::buffer_type::resizable);
    cuda_cpy.setup(recorded_intersections_buffer)->wait();
    auto recorded_intersections_view =
        vecmem::get_data(recorded_intersections_buffer);

    vecmem::data::vector_buffer<material_record_t> mat_records_buffer(
        static_cast<unsigned int>(truth_intersection_traces_view.size()),
        *dev_mr, vecmem::data::buffer_type::fixed_size);
    cuda_cpy.setup(mat_records_buffer)->wait();
    auto mat_records_view = vecmem::get_data(mat_records_buffer);

    // Buffer for the material parameters at every step per track
    vecmem::data::jagged_vector_buffer<material_params_t> mat_steps_buffer(
        capacities, *dev_mr, host_mr, vecmem::data::buffer_type::resizable);
    cuda_cpy.setup(mat_steps_buffer)->wait();
    auto mat_steps_view = vecmem::get_data(mat_steps_buffer);

    // Run the navigation validation test on device
    navigation_validation_device<bfield_t, detector_t, intersection_record_t>(
        det_view, cfg, field_data, truth_intersection_traces_view,
        recorded_intersections_view, mat_records_view, mat_steps_view);

    // Get the results back to the host and pass them on to the checking
    vecmem::jagged_vector<navigation::detail::candidate_record<intersection_t>>
        recorded_intersections(host_mr);
    cuda_cpy(recorded_intersections_buffer, recorded_intersections)->wait();

    vecmem::vector<material_record_t> mat_records(host_mr);
    cuda_cpy(mat_records_buffer, mat_records)->wait();

    vecmem::jagged_vector<material_params_t> mat_steps(host_mr);
    cuda_cpy(mat_steps_buffer, mat_steps)->wait();

    return std::make_tuple(std::move(recorded_intersections),
                           std::move(mat_records), std::move(mat_steps));
}

/// @brief Test class that runs the navigation validation for a given detector
/// on device.
///
/// @note The lifetime of the detector needs to be guaranteed outside this class
template <typename detector_t, template <typename> class scan_type>
class navigation_validation : public test::fixture_base<> {

    using scalar_t = typename detector_t::scalar_type;
    using algebra_t = typename detector_t::algebra_type;
    using vector3_t = typename detector_t::vector3_type;
    using free_track_parameters_t = free_track_parameters<algebra_t>;
    using trajectory_type = typename scan_type<algebra_t>::trajectory_type;
    using intersection_trace_t = typename scan_type<
        algebra_t>::template intersection_trace_type<detector_t>;

    /// Switch between rays and helices
    static constexpr auto k_use_rays{
        std::is_same_v<detail::ray<algebra_t>, trajectory_type>};

    public:
    using fixture_type = test::fixture_base<>;
    using config = detray::test::navigation_validation_config;

    explicit navigation_validation(
        const detector_t &det, const typename detector_t::name_map &names,
        const config &cfg = {},
        const typename detector_t::geometry_context gctx = {})
        : m_cfg{cfg}, m_gctx{gctx}, m_det{det}, m_names{names} {

        if (!m_cfg.whiteboard()) {
            throw std::invalid_argument("No white board was passed to " +
                                        m_cfg.name() + " test");
        }

        // Use ray or helix
        const std::string det_name{m_det.name(m_names)};
        m_truth_data_name = k_use_rays ? det_name + "_ray_scan_for_cuda"
                                       : det_name + "_helix_scan_for_cuda";

        // Pin the data onto the whiteboard
        if (!m_cfg.whiteboard()->exists(m_truth_data_name) &&
            io::file_exists(m_cfg.intersection_file()) &&
            io::file_exists(m_cfg.track_param_file())) {

            // Name clash: Choose alternative name
            if (m_cfg.whiteboard()->exists(m_truth_data_name)) {
                m_truth_data_name = io::alt_file_name(m_truth_data_name);
            }

            std::vector<intersection_trace_t> intersection_traces;

            std::cout << "\nINFO: Reading data from file..." << std::endl;

            // Fill the intersection traces from file
            detray::detector_scanner::read(m_cfg.intersection_file(),
                                           m_cfg.track_param_file(),
                                           intersection_traces);

            m_cfg.whiteboard()->add(m_truth_data_name,
                                    std::move(intersection_traces));
        } else if (m_cfg.whiteboard()->exists(m_truth_data_name)) {
            std::cout << "\nINFO: Fetching data from white board..."
                      << std::endl;
        } else {
            throw std::invalid_argument(
                "Navigation validation: Could not find data files");
        }

        // Check that data is ready
        if (!m_cfg.whiteboard()->exists(m_truth_data_name)) {
            throw std::invalid_argument(
                "Data for navigation check is not on the whiteboard");
        }
    }

    /// Run the check
    void TestBody() override {
        using namespace detray;
        using namespace navigation;

        // Runge-Kutta stepper
        using hom_bfield_t = bfield::const_field_t;
        using bfield_view_t =
            std::conditional_t<k_use_rays, navigation_validator::empty_bfield,
                               hom_bfield_t::view_t>;
        using bfield_t =
            std::conditional_t<k_use_rays, navigation_validator::empty_bfield,
                               hom_bfield_t>;
        using intersection_t =
            typename intersection_trace_t::value_type::intersection_type;

        bfield_t b_field{};
        if constexpr (!k_use_rays) {
            b_field = bfield::create_const_field(m_cfg.B_vector());
        }

        // Fetch the truth data
        auto &truth_intersection_traces =
            m_cfg.whiteboard()->template get<std::vector<intersection_trace_t>>(
                m_truth_data_name);

        std::size_t n_test_tracks{
            std::min(m_cfg.n_tracks(), truth_intersection_traces.size())};
        std::cout << "\nINFO: Running device navigation validation on: "
                  << m_det.name(m_names) << "...\n"
                  << std::endl;

        const std::string det_name{m_det.name(m_names)};
        const std::string prefix{k_use_rays ? det_name + "_ray_"
                                            : det_name + "_helix_"};

        std::ios_base::openmode io_mode = std::ios::trunc | std::ios::out;
        const std::string debug_file_name{prefix +
                                          "navigation_validation_cuda.txt"};
        detray::io::file_handle debug_file{debug_file_name, io_mode};

        // Run the propagation on device and record the navigation data
        auto [recorded_intersections, mat_records, mat_steps] =
            run_navigation_validation<bfield_view_t>(
                &m_host_mr, &m_dev_mr, m_det, m_cfg.propagation(), b_field,
                truth_intersection_traces);

        // Collect some statistics
        std::size_t n_tracks{0u};
        std::size_t n_surfaces{0u};
        std::size_t n_miss_nav{0u};
        std::size_t n_miss_truth{0u};
        std::size_t n_matching_error{0u};
        std::size_t n_fatal{0u};

        std::vector<std::pair<trajectory_type, std::vector<intersection_t>>>
            missed_intersections{};

        EXPECT_EQ(recorded_intersections.size(),
                  truth_intersection_traces.size());

        scalar_t min_pT{std::numeric_limits<scalar_t>::max()};
        scalar_t max_pT{-std::numeric_limits<scalar_t>::max()};
        for (std::size_t i = 0u; i < truth_intersection_traces.size(); ++i) {
            auto &truth_trace = truth_intersection_traces[i];
            auto &recorded_trace = recorded_intersections[i];

            if (n_tracks >= m_cfg.n_tracks()) {
                break;
            }

            // Get the original test trajectory (ray or helix)
            const auto &start = truth_trace.front();
            const auto &trck_param = start.track_param;
            trajectory_type test_traj = get_parametrized_trajectory(trck_param);

            const scalar q = start.charge;
            const scalar pT{q == 0.f ? 1.f * unit<scalar>::GeV
                                     : trck_param.pT(q)};
            const scalar p{q == 0.f ? 1.f * unit<scalar>::GeV
                                    : trck_param.p(q)};
            min_pT = std::min(min_pT, pT);
            max_pT = std::max(max_pT, pT);

            // Recorded only the start position, which added by default
            bool success{true};
            if (truth_trace.size() == 1) {
                // Propagation did not succeed
                success = false;
                std::vector<intersection_t> missed_inters{};
                missed_intersections.push_back(
                    std::make_pair(test_traj, missed_inters));

                ++n_fatal;
            } else {
                // Adjust the track charge, which is unknown to the navigation
                for (auto &record : recorded_trace) {
                    record.charge = q;
                    record.p_mag = p;
                }

                // Compare truth and recorded data elementwise
                auto [result, n_missed_nav, n_missed_truth, n_error,
                      missed_inters] =
                    navigation_validator::compare_traces(
                        truth_trace, recorded_trace, test_traj, n_tracks,
                        n_test_tracks, &(*debug_file));

                missed_intersections.push_back(
                    std::make_pair(test_traj, std::move(missed_inters)));

                // Update statistics
                success &= result;
                n_miss_nav += n_missed_nav;
                n_miss_truth += n_missed_truth;
                n_matching_error += n_error;
            }

            if (!success) {
                detector_scanner::display_error(
                    m_gctx, m_det, m_names, m_cfg.name(), test_traj,
                    truth_trace, m_cfg.svg_style(), n_tracks, n_test_tracks,
                    recorded_trace);
            }

            EXPECT_TRUE(success) << "INFO: Wrote navigation debugging data in: "
                                 << debug_file_name;

            ++n_tracks;

            ASSERT_EQ(truth_trace.size(), recorded_trace.size());
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
        const auto truth_trk_path{
            data_path /
            (prefix + "truth_track_params_cuda_" + mometum_str + "GeV.csv")};
        const auto trk_path{data_path /
                            (prefix + "navigation_track_params_cuda_" +
                             mometum_str + "GeV.csv")};
        const auto mat_path{data_path / (prefix + "accumulated_material_cuda_" +
                                         mometum_str + "GeV.csv")};
        const auto missed_path{data_path /
                               (prefix + "missed_intersections_dists_cuda_" +
                                mometum_str + "GeV.csv")};

        // Write the distance of the missed intersection local position
        // to the surface boundaries to file for plotting
        navigation_validator::write_dist_to_boundary(
            m_det, m_names, missed_path.string(), missed_intersections);
        detector_scanner::write_tracks(truth_trk_path.string(),
                                       truth_intersection_traces);
        navigation_validator::write_tracks(trk_path.string(),
                                           recorded_intersections);
        material_validator::write_material(mat_path.string(), mat_records);

        std::cout
            << "INFO: Wrote distance to boundary of missed intersections to: "
            << missed_path << std::endl;
        std::cout << "INFO: Wrote track states in: " << trk_path << std::endl;
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

    /// Vecmem memory resource for the host allocations
    vecmem::host_memory_resource m_host_mr{};
    /// Vecmem memory resource for the device allocations
    vecmem::cuda::device_memory_resource m_dev_mr{};
    /// The configuration of this test
    config m_cfg;
    /// Name of the truth data collection
    std::string m_truth_data_name{""};
    /// The geometry context to check
    typename detector_t::geometry_context m_gctx{};
    /// The detector to be checked
    const detector_t &m_det;
    /// Volume names
    const typename detector_t::name_map &m_names;
};

template <typename detector_t>
using straight_line_navigation =
    detray::cuda::navigation_validation<detector_t, detray::ray_scan>;

template <typename detector_t>
using helix_navigation =
    detray::cuda::navigation_validation<detector_t, detray::helix_scan>;

}  // namespace detray::cuda
