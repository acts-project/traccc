/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/core/detector.hpp"
#include "detray/tracks/tracks.hpp"
#include "detray/utils/ranges.hpp"

// Detray test include(s)
#include "detray/test/common/fixture_base.hpp"
#include "detray/test/common/material_validation_config.hpp"
#include "detray/test/validation/material_validation_utils.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s)
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace detray::test {

/// Run the material validation on the host
struct run_material_validation {

    static constexpr std::string_view name{"cpu"};

    template <typename detector_t>
    auto operator()(
        vecmem::memory_resource *host_mr, vecmem::memory_resource *,
        const detector_t &det, const propagation::config &cfg,
        const dvector<free_track_parameters<typename detector_t::algebra_type>>
            &tracks,
        const std::vector<std::size_t> & = {}) {

        using scalar_t = typename detector_t::scalar_type;

        typename detector_t::geometry_context gctx{};

        dvector<material_validator::material_record<scalar_t>> mat_records{
            host_mr};
        mat_records.reserve(tracks.size());

        dvector<dvector<material_validator::material_params<scalar_t>>>
            mat_steps_vec{host_mr};

        mat_steps_vec.reserve(tracks.size());

        for (const auto &[i, track] : detray::views::enumerate(tracks)) {

            auto [success, mat_record, mat_steps] =
                detray::material_validator::record_material(gctx, host_mr, det,
                                                            cfg, track);
            mat_records.push_back(mat_record);
            mat_steps_vec.push_back(std::move(mat_steps));

            if (!success) {
                std::cerr << "ERROR: Propagation failed for track " << i << ": "
                          << "Material record may be incomplete!" << std::endl;
            }
        }

        return std::make_tuple(std::move(mat_records),
                               std::move(mat_steps_vec));
    }
};

/// @brief Test class that runs the material validation for a given detector.
///
/// @note The lifetime of the detector needs to be guaranteed outside this class
template <typename detector_t, typename material_validator_t>
class material_validation_impl : public test::fixture_base<> {

    using scalar_t = typename detector_t::scalar_type;
    using algebra_t = typename detector_t::algebra_type;
    using free_track_parameters_t = free_track_parameters<algebra_t>;
    using material_record_t = material_validator::material_record<scalar_t>;

    public:
    using fixture_type = test::fixture_base<>;
    using config = detray::test::material_validation_config;

    explicit material_validation_impl(
        const detector_t &det, const typename detector_t::name_map &names,
        const config &cfg = {},
        const typename detector_t::geometry_context gctx = {})
        : m_cfg{cfg}, m_gctx{gctx}, m_det{det}, m_names{names} {

        if (!m_cfg.whiteboard()) {
            throw std::invalid_argument("No white board was passed to " +
                                        m_cfg.name() + " test");
        }

        // Name of the material scan data collection
        m_scan_data_name = m_det.name(m_names) + "_material_scan";
        m_track_data_name = m_det.name(m_names) + "_material_scan_tracks";

        // Check that data is available in memory
        if (!m_cfg.whiteboard()->exists(m_scan_data_name)) {
            throw std::invalid_argument(
                "Material validation: Could not find scan data on whiteboard."
                "Please run material scan first.");
        }
        if (!m_cfg.whiteboard()->exists(m_track_data_name)) {
            throw std::invalid_argument(
                "Material validation: Could not find track data on whiteboard."
                "Please run material scan first.");
        }
    }

    /// Run the check
    void TestBody() override {
        using namespace detray;

        // Fetch the input data
        const auto &tracks =
            m_cfg.whiteboard()->template get<dvector<free_track_parameters_t>>(
                m_track_data_name);

        const auto &truth_mat_records =
            m_cfg.whiteboard()->template get<dvector<material_record_t>>(
                m_scan_data_name);

        std::cout << "\nINFO: Running material validation on: "
                  << m_det.name(m_names) << "...\n"
                  << std::endl;

        // only needed for device material steps allocations
        // @TODO: For now, guess how many surface might be encountered
        std::vector<std::size_t> capacities(tracks.size(), 80u);

        // Run the propagation on device and record the accumulated material
        auto [mat_records, mat_steps] =
            material_validator_t{}(&m_host_mr, m_cfg.device_mr(), m_det,
                                   m_cfg.propagation(), tracks, capacities);

        // One material record per track
        ASSERT_EQ(tracks.size(), mat_records.size());

        // Collect some statistics
        std::size_t n_tracks{0u};
        const scalar_t rel_error{m_cfg.relative_error()};
        for (std::size_t i = 0u; i < mat_records.size(); ++i) {

            if (n_tracks >= m_cfg.n_tracks()) {
                break;
            }

            const auto &truth_mat = truth_mat_records[i];
            const auto &recorded_mat = mat_records[i];

            auto get_rel_error = [](const scalar_t truth, const scalar_t rec) {
                constexpr scalar_t e{std::numeric_limits<scalar_t>::epsilon()};

                if (truth <= e && rec <= e) {
                    // No material for this ray => valid
                    return scalar_t{0.f};
                } else if (truth <= e) {
                    // Material found where none should be
                    return detail::invalid_value<scalar_t>();
                } else {
                    return math::fabs(truth - rec) / truth;
                }
            };

            EXPECT_TRUE(get_rel_error(truth_mat.sX0, recorded_mat.sX0) <
                        rel_error)
                << "Track " << n_tracks << " (X0 / path): Truth "
                << truth_mat.sX0 << ", Nav. " << recorded_mat.sX0;
            EXPECT_TRUE(get_rel_error(truth_mat.tX0, recorded_mat.tX0) <
                        rel_error)
                << "Track " << n_tracks << " (X0 / thickness): Truth "
                << truth_mat.tX0 << ", Nav. " << recorded_mat.tX0;
            EXPECT_TRUE(get_rel_error(truth_mat.sL0, recorded_mat.sL0) <
                        rel_error)
                << "Track " << n_tracks << " (L0 / path): Truth "
                << truth_mat.sL0 << ", Nav. " << recorded_mat.sL0;
            EXPECT_TRUE(get_rel_error(truth_mat.tL0, recorded_mat.tL0) <
                        rel_error)
                << "Track " << n_tracks << " (L0 / thickness): Truth "
                << truth_mat.tL0 << ", Nav. " << recorded_mat.tL0;

            ++n_tracks;
        }

        std::cout << "-----------------------------------\n"
                  << "Tested " << n_tracks << " tracks\n"
                  << "-----------------------------------\n"
                  << std::endl;

        // Print accumulated material per track
        std::filesystem::path mat_path{m_cfg.material_file()};
        auto file_name{m_det.name(m_names) + "_" +
                       std::string(material_validator_t::name) + "_" +
                       mat_path.stem().string() +
                       mat_path.extension().string()};

        material_validator::write_material(mat_path.replace_filename(file_name),
                                           mat_records);
    }

    private:
    /// The configuration of this test
    config m_cfg;
    /// Vecmem memory resource for the host allocations
    vecmem::host_memory_resource m_host_mr{};
    /// Name of the input data collections
    std::string m_scan_data_name{""};
    std::string m_track_data_name{""};
    /// The geometry context to check
    typename detector_t::geometry_context m_gctx{};
    /// The detector to be checked
    const detector_t &m_det;
    /// Volume names
    const typename detector_t::name_map &m_names;
};

template <typename detector_t>
using material_validation =
    material_validation_impl<detector_t, run_material_validation>;

}  // namespace detray::test
