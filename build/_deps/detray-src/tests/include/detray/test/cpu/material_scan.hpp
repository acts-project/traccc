/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/navigation/detail/ray.hpp"

// Detray IO include(s)
#include "detray/io/utils/file_handle.hpp"

// Detray test include(s)
#include "detray/test/common/detail/whiteboard.hpp"
#include "detray/test/common/fixture_base.hpp"
#include "detray/test/utils/simulation/event_generator/track_generators.hpp"
#include "detray/test/utils/types.hpp"
#include "detray/test/validation/detector_scanner.hpp"
#include "detray/test/validation/material_validation_utils.hpp"

// System include(s)
#include <ios>
#include <iostream>
#include <string>

namespace detray::test {

/// @brief Test class that runs the material ray scan on a given detector.
///
/// @note The lifetime of the detector needs to be guaranteed.
template <typename detector_t>
class material_scan : public test::fixture_base<> {

    using algebra_t = typename detector_t::algebra_type;
    using point2_t = typename detector_t::point2_type;
    using scalar_t = typename detector_t::scalar_type;
    using ray_t = detail::ray<algebra_t>;
    using track_generator_t = uniform_track_generator<ray_t>;

    public:
    using fixture_type = test::fixture_base<>;

    struct config : public fixture_type::configuration {
        using trk_gen_config_t = typename track_generator_t::configuration;

        std::string m_name{"material_scan"};
        /// Save results for later use in downstream tests
        std::shared_ptr<test::whiteboard> m_white_board;
        trk_gen_config_t m_trk_gen_cfg{};

        /// Getters
        /// @{
        const std::string &name() const { return m_name; }
        trk_gen_config_t &track_generator() { return m_trk_gen_cfg; }
        const trk_gen_config_t &track_generator() const {
            return m_trk_gen_cfg;
        }
        std::shared_ptr<test::whiteboard> whiteboard() { return m_white_board; }
        std::shared_ptr<test::whiteboard> whiteboard() const {
            return m_white_board;
        }
        /// @}

        /// Setters
        /// @{
        config &name(const std::string n) {
            m_name = n;
            return *this;
        }
        config &whiteboard(std::shared_ptr<test::whiteboard> w_board) {
            if (!w_board) {
                throw std::invalid_argument(
                    "Material scan: Not a valid whiteboard instance");
            }
            m_white_board = std::move(w_board);
            return *this;
        }
        /// @}
    };

    explicit material_scan(
        const detector_t &det, const typename detector_t::name_map &names,
        const config &cfg = {},
        const typename detector_t::geometry_context &gctx = {})
        : m_cfg{cfg}, m_gctx{gctx}, m_det{det}, m_names{names} {

        if (!m_cfg.whiteboard()) {
            throw std::invalid_argument("No white board was passed to " +
                                        m_cfg.name() + " test");
        }
    }

    /// Run the ray scan
    void TestBody() override {

        using nav_link_t = typename detector_t::surface_type::navigation_link;
        using material_record_t = material_validator::material_record<scalar_t>;

        std::size_t n_tracks{0u};
        auto ray_generator = track_generator_t(m_cfg.track_generator());

        std::cout << "INFO: Running material scan on: " << m_det.name(m_names)
                  << "\n(" << ray_generator.size() << " rays) ...\n"
                  << std::endl;

        // Trace material per ray
        dvector<free_track_parameters<algebra_t>> tracks{};
        tracks.reserve(ray_generator.size());

        dvector<material_record_t> mat_records{};
        mat_records.reserve(ray_generator.size());

        for (const auto ray : ray_generator) {

            // Record all intersections and surfaces along the ray
            const auto intersection_record =
                detector_scanner::run<detray::ray_scan>(m_gctx, m_det, ray);

            if (intersection_record.empty()) {
                std::cout << "ERROR: Intersection trace empty for ray "
                          << n_tracks << "/" << ray_generator.size() << ": "
                          << ray << std::endl;
                break;
            }

            // Record track parameters
            tracks.push_back({ray.pos(), 0.f, ray.dir(), -1.f});

            // New material record
            material_record_t mat_record{};
            mat_record.eta = getter::eta(ray.dir());
            mat_record.phi = getter::phi(ray.dir());

            // Record material for this ray
            for (const auto &[i, record] :
                 detray::views::enumerate(intersection_record)) {

                // Prevent double counting of material on adjacent portals
                if ((i < intersection_record.size() + 1) &&
                    (((intersection_record[i + 1].intersection ==
                       record.intersection) &&
                      intersection_record[i + 1]
                          .intersection.sf_desc.is_portal() &&
                      record.intersection.sf_desc.is_portal()) ||
                     (record.intersection.volume_link ==
                      detail::invalid_value<nav_link_t>()))) {
                    continue;
                }

                const auto sf =
                    tracking_surface{m_det, record.intersection.sf_desc};

                if (!sf.has_material()) {
                    continue;
                }

                const auto &p = record.intersection.local;
                const auto mat_params = sf.template visit_material<
                    material_validator::get_material_params>(
                    point2_t{p[0], p[1]}, sf.cos_angle(m_gctx, ray.dir(), p));

                const scalar_t seg{mat_params.path};
                const scalar_t t{mat_params.thickness};
                const scalar_t mx0{mat_params.mat_X0};
                const scalar_t ml0{mat_params.mat_L0};

                if (mx0 > 0.f) {
                    mat_record.sX0 += seg / mx0;
                    mat_record.tX0 += t / mx0;
                } else {
                    std::cout << "WARNING: Encountered invalid X_0: " << mx0
                              << "\nOn surface: " << sf << std::endl;
                }
                if (ml0 > 0.f) {
                    mat_record.sL0 += seg / ml0;
                    mat_record.tL0 += t / ml0;
                } else {
                    std::cout << "WARNING: Encountered invalid L_0: " << ml0
                              << "\nOn surface: " << sf << std::endl;
                }
            }

            if (mat_record.sX0 == 0.f || mat_record.sL0 == 0.f ||
                mat_record.tX0 == 0.f || mat_record.tL0 == 0.f) {
                std::cout << "WARNING: No material recorded for ray "
                          << n_tracks << "/" << ray_generator.size() << ": "
                          << ray << std::endl;
            }

            mat_records.push_back(mat_record);

            ++n_tracks;
        }

        std::cout << "-----------------------------------\n"
                  << "Tested " << n_tracks << " tracks\n"
                  << "-----------------------------------\n"
                  << std::endl;

        // Write recorded material to csv file
        std::string coll_name{m_det.name(m_names) + "_material_scan"};
        material_validator::write_material(coll_name + ".csv", mat_records);

        // Pin data to whiteboard
        m_cfg.whiteboard()->add(coll_name, std::move(mat_records));
        m_cfg.whiteboard()->add(m_det.name(m_names) + "_material_scan_tracks",
                                std::move(tracks));
    }

    private:
    /// The configuration of this test
    config m_cfg;
    /// The geometry context to scan
    typename detector_t::geometry_context m_gctx{};
    /// The detector to be checked
    const detector_t &m_det;
    /// Volume names
    const typename detector_t::name_map &m_names;
};

}  // namespace detray::test
