/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../common/make_magnetic_field.hpp"
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/propagation.hpp"

// io
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/io/utils.hpp"

// algorithms
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// performance
#include "traccc/efficiency/finding_performance_writer.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"

// options
#include <detray/test/common/track_generators.hpp>
#include <detray/test/validation/detector_scanner.hpp>

#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/magnetic_field.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_resolution.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/options/truth_finding.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <cassert>
#include <cstdlib>
#include <iostream>

using namespace traccc;

int run_detector_scan(const traccc::opts::track_seeding& seeding_opts,
                      const traccc::opts::detector& detector_opts,
                      std::unique_ptr<const traccc::Logger> ilogger) {
    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    seedfinder_config seeding_conf(seeding_opts);

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Copy obejct
    vecmem::copy copy;

    using detector_t = traccc::default_detector::host;
    using ray_t = detray::detail::ray<detector_t::algebra_type>;

    // Construct a Detray detector object, if supported by the configuration.
    detector_t detector{host_mr};
    assert(detector_opts.use_detray_detector == true);
    traccc::io::read_detector(detector, host_mr, detector_opts.detector_file,
                              detector_opts.material_file,
                              detector_opts.grid_file);

    const detector_t::geometry_context gctx{};

    std::map<std::pair<std::size_t, std::size_t>,
             std::tuple<float, float, float, float>>
        mapping;

    for (float phi = 0; phi < 6.283185307f; phi += 8.f) {
        for (float z0 = -150.f; z0 <= 150.f; z0 += 10.f) {
            for (float theta = 0; theta <= 1.57079633; theta += 0.01f) {
                std::cout << z0 << ", " << theta << std::endl;
                float sin_theta = std::sin(theta);
                detray::dvector3D<detector_t::algebra_type> p{
                    math::cos(phi) * sin_theta, math::sin(phi) * sin_theta,
                    math::cos(theta)};

                p = vector::normalize(p);

                ray_t ray{{0.f, 0.f, z0}, 0.f, p, 1};

                // Record all intersections and surfaces along the ray
                const auto intersection_trace =
                    detray::detector_scanner::run<detray::ray_scan>(
                        gctx, detector, ray,
                        detray::darray<float, 2>{5.f, 5.f});

                std::vector<unsigned int> sensitive_idxs;
                unsigned int q = 0;
                for (const auto& i : intersection_trace) {
                    const auto& p = i.track_param;
                    const float r = std::hypot(p.pos()[0], p.pos()[1]);

                    if (i.intersection.sf_desc.id() ==
                            detray::surface_id::e_sensitive &&
                        r <= seeding_conf.rMax && r >= seeding_conf.rMin &&
                        p.pos()[2] <= seeding_conf.zMax &&
                        p.pos()[2] >= seeding_conf.zMin) {
                        sensitive_idxs.push_back(q);
                    }
                    q++;
                }

                unsigned int seed_length = 3;

                if (sensitive_idxs.size() < seed_length)
                    continue;

                unsigned int shortest_chain_idx = 0;
                unsigned int shortest_chain_length = 0;
                float shortest_chain_weight = 10000000000.f;

                for (unsigned int i = 0;
                     i < sensitive_idxs.size() - seed_length; i++) {
                    unsigned int j = static_cast<std::size_t>(i + seed_length);

                    unsigned ii = sensitive_idxs.at(i);
                    unsigned ij = sensitive_idxs.at(j);

                    const auto& si = intersection_trace.at(ii).intersection;
                    const auto& sj = intersection_trace.at(ij).intersection;

                    const auto& pi = intersection_trace.at(ii).track_param;
                    const auto& pj = intersection_trace.at(ij).track_param;

                    const float ri = std::hypot(pi.pos()[0], pi.pos()[1]);
                    const float rj = std::hypot(pj.pos()[0], pj.pos()[1]);

                    float weight =
                        std::hypot(rj - ri, pj.pos()[2] - pi.pos()[2]);

                    if (weight < shortest_chain_weight) {
                        shortest_chain_idx = i;
                        shortest_chain_length = j - i;
                        shortest_chain_weight = weight;
                    }
                }

                for (unsigned int i = 0; i < shortest_chain_length; i++) {
                    unsigned int j = i + 1;

                    unsigned ii = sensitive_idxs.at(shortest_chain_idx + i);
                    unsigned ij = sensitive_idxs.at(shortest_chain_idx + j);

                    const auto& si = intersection_trace.at(ii).intersection;
                    const auto& sj = intersection_trace.at(ij).intersection;

                    const auto& pi = intersection_trace.at(ii).track_param;
                    const auto& pj = intersection_trace.at(ij).track_param;

                    const float ri = std::hypot(pi.pos()[0], pi.pos()[1]);
                    const float rj = std::hypot(pj.pos()[0], pj.pos()[1]);

                    std::size_t di = si.sf_desc.index();
                    std::size_t dj = sj.sf_desc.index();

                    if (di == dj)
                        continue;

                    if (auto it = mapping.find({di, dj}); it != mapping.end()) {
                        std::get<0>(it->second) =
                            std::min(std::get<0>(it->second), ri);
                        std::get<1>(it->second) = std::min(
                            std::get<1>(it->second), std::abs(pi.pos()[2]));
                        std::get<2>(it->second) =
                            std::max(std::get<2>(it->second), (rj - ri));
                        std::get<3>(it->second) =
                            std::max(std::get<3>(it->second),
                                     std::abs(pj.pos()[2] - pi.pos()[2]));
                    } else {
                        mapping[{di, dj}] = {
                            ri, std::abs(pi.pos()[2]), rj - ri,
                            std::abs(pj.pos()[2] - pi.pos()[2])};
                    }
                }
            }
        }
    }

    std::cout << mapping.size() << std::endl;

    std::cout << "Pareto set for delta r" << std::endl;

    unsigned int idx1 = 0;
    for (const auto& it1 : mapping) {
        const auto& t1 = it1.second;
        bool is_dominated = false;

        unsigned int idx2 = 0;
        for (const auto& it2 : mapping) {
            if (it1 == it2)
                continue;

            const auto& t2 = it2.second;

            if (std::get<0>(t2) <= std::get<0>(t1) &&
                // std::get<1>(t2) <= std::get<1>(t1) &&
                std::get<2>(t2) >= std::get<2>(t1)
                // std::get<3>(t2) >= std::get<3>(t1)
            ) {
                if (std::get<0>(t2) == std::get<0>(t1) &&
                    std::get<2>(t2) == std::get<2>(t1)) {
                    is_dominated = idx2 < idx1;
                } else {
                    is_dominated = true;
                }
            }

            idx2++;

            if (is_dominated)
                break;
        }

        if (!is_dominated) {
            std::cout  //<< it1.first.first << ", " << it1.first.second << ", "
                << std::get<0>(it1.second)
                << ","
                //<< std::get<1>(it1.second) << ", "
                << std::get<2>(it1.second)  // << ", "
                //<< std::get<3>(it1.second)
                << std::endl;
        }

        idx1++;
    }

    std::cout << "Pareto set for delta z" << std::endl;

    idx1 = 0;
    for (const auto& it1 : mapping) {
        const auto& t1 = it1.second;
        bool is_dominated = false;

        unsigned int idx2 = 0;
        for (const auto& it2 : mapping) {
            if (it1 == it2)
                continue;

            const auto& t2 = it2.second;

            if (  // std::get<0>(t2) <= std::get<0>(t1) &&
                std::get<1>(t2) <= std::get<1>(t1) &&
                // std::get<2>(t2) >= std::get<2>(t1)
                std::get<3>(t2) >= std::get<3>(t1)) {
                if (std::get<1>(t2) == std::get<1>(t1) &&
                    std::get<3>(t2) == std::get<3>(t1)) {
                    is_dominated = idx2 < idx1;
                } else {
                    is_dominated = true;
                }
            }

            idx2++;

            if (is_dominated)
                break;
        }

        if (!is_dominated) {
            std::cout  //<< it1.first.first << ", " << it1.first.second << ", "
                       //<< std::get<0>(it1.second) << ", "
                << std::get<1>(it1.second)
                << ","
                //<< std::get<2>(it1.second) << ", "
                << std::get<3>(it1.second) << std::endl;
        }

        idx1++;
    }

    return EXIT_SUCCESS;
}

// The main routine
//
int main(int argc, char* argv[]) {
    std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
        "TracccDetectorScan", traccc::Logging::Level::INFO);

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::track_seeding seeding_opts;
    traccc::opts::program_options program_opts{
        "Full Tracking Chain on the Host (without clusterization)",
        {detector_opts, seeding_opts},
        argc,
        argv,
        logger->cloneWithSuffix("Options")};

    // Run the application.
    return run_detector_scan(seeding_opts, detector_opts, logger->clone());
}
