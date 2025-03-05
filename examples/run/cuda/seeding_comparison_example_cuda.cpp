/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/accelerator.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/container_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/utils/seed_generator.hpp"

// detray include(s).
#include <detray/detectors/bfield.hpp>
#include <detray/io/frontend/detector_reader.hpp>
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/propagator.hpp>
#include <detray/propagator/rk_stepper.hpp>

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>

struct output_data_point {
    std::size_t ptc_id;
    bool found_in_truth = false;
    bool found_in_reco = false;
    float bot_x, bot_y, bot_z;
    float mid_x, mid_y, mid_z;
    float top_x, top_y, top_z;
};

using namespace traccc;
int seq_run(const traccc::opts::input_data& input_opts,
            const traccc::opts::detector& detector_opts,
            const traccc::opts::track_seeding& seeding_opts,
            std::unique_ptr<const traccc::Logger> ilogger) {
    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::managed_memory_resource mng_mr;
    traccc::memory_resource mr{mng_mr, &cuda_host_mr};

    // Construct a Detray detector object, if supported by the configuration.
    traccc::default_detector::host detector{mng_mr};
    assert(detector_opts.use_detray_detector == true);
    traccc::io::read_detector(detector, mng_mr, detector_opts.detector_file,
                              detector_opts.material_file,
                              detector_opts.grid_file);

    // Detector view object
    [[maybe_unused]] traccc::default_detector::view det_view =
        detray::get_data(detector);

    /*****************************
     * Do the reconstruction
     *****************************/

    // Stream object
    traccc::cuda::stream stream;

    // Copy object
    vecmem::cuda::async_copy async_copy{stream.cudaStream()};

    traccc::cuda::seeding_algorithm sa_cuda{seeding_opts.seedfinder,
                                            {seeding_opts.seedfinder},
                                            seeding_opts.seedfilter,
                                            mr,
                                            async_copy,
                                            stream,
                                            logger().clone("CudaSeedingAlg")};
    traccc::device::container_d2h_copy_alg<
        traccc::track_candidate_container_types>
        track_candidate_d2h{mr, async_copy,
                            logger().clone("TrackCandidateD2HCopyAlg")};

    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        track_state_d2h{mr, async_copy, logger().clone("TrackStateD2HCopyAlg")};

    // Standard deviations for seed track parameters
    static constexpr std::array<traccc::scalar, traccc::e_bound_size> stddevs =
        {1e-4f * traccc::unit<traccc::scalar>::mm,
         1e-4f * traccc::unit<traccc::scalar>::mm,
         1e-3f,
         1e-3f,
         1e-4f / traccc::unit<traccc::scalar>::GeV,
         1e-4f * traccc::unit<traccc::scalar>::ns};

    traccc::seed_generator<traccc::default_detector::host> sg(detector,
                                                              stddevs);
    // Iterate over events
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {
        // Truth Track Candidates
        traccc::event_data evt_data(input_opts.directory, event, host_mr,
                                    input_opts.use_acts_geom_source, &detector,
                                    input_opts.format, false);

        traccc::track_candidate_container_types::host truth_track_candidates =
            evt_data.generate_truth_candidates(sg, host_mr,
                                               0.5 / unit<float>::GeV);
        traccc::edm::spacepoint_collection::host spacepoints_per_event{host_mr};
        traccc::measurement_collection_types::host measurements_per_event{
            &host_mr};
        traccc::io::read_spacepoints(
            spacepoints_per_event, measurements_per_event, event,
            input_opts.directory,
            (input_opts.use_acts_geom_source ? &detector : nullptr),
            input_opts.format);

        std::map<measurement, std::size_t>
            measurement_to_evt_data_measurement_map;
        std::size_t num_unmatched_measurements = 0;

        for (std::size_t i = 0; i < measurements_per_event.size(); ++i) {
            bool matched = false;

            for (const auto& kv : evt_data.m_measurement_map) {
                if (kv.second == measurements_per_event.at(i)) {
                    measurement_to_evt_data_measurement_map
                        [measurements_per_event.at(i)] = kv.first;
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                num_unmatched_measurements++;
            }
        }

        if (num_unmatched_measurements > 0) {
            TRACCC_WARNING("At least " << num_unmatched_measurements
                                       << " measurements are unmatched");
        } else {
            TRACCC_INFO("All measurements are matched; nice!");
        }

        std::vector<output_data_point> output_data_points;
        std::map<std::tuple<measurement, measurement, measurement>, std::size_t>
            seed_to_output_data_point_idx_map;

        // === TRUTH SEEDING ===
        std::size_t num_truth_tracks = 0;

        {
            std::size_t num_untrackable_particles = 0;

            // Prepare truth seeds
            for (std::size_t i_trk = 0; i_trk < truth_track_candidates.size();
                 i_trk++) {
                if (truth_track_candidates.at(i_trk).items.size() < 3) {
                    num_untrackable_particles++;
                    continue;
                }

                const measurement& bot_meas =
                    truth_track_candidates.at(i_trk).items.at(0);
                const measurement& mid_meas =
                    truth_track_candidates.at(i_trk).items.at(1);
                const measurement& top_meas =
                    truth_track_candidates.at(i_trk).items.at(2);

                const std::map<particle, std::size_t>& bot_ptc_map =
                    evt_data.m_meas_to_ptc_map.at(bot_meas);
                const std::map<particle, std::size_t>& mid_ptc_map =
                    evt_data.m_meas_to_ptc_map.at(mid_meas);
                const std::map<particle, std::size_t>& top_ptc_map =
                    evt_data.m_meas_to_ptc_map.at(top_meas);

                std::vector<particle> intersection;

                for (const auto& it_bot : bot_ptc_map) {
                    for (const auto& it_mid : mid_ptc_map) {
                        for (const auto& it_top : top_ptc_map) {
                            if (it_bot.first.particle_id ==
                                    it_mid.first.particle_id &&
                                it_mid.first.particle_id ==
                                    it_top.first.particle_id) {
                                intersection.push_back(it_bot.first);
                            }
                        }
                    }
                }

                if (intersection.size() != 1) {
                    TRACCC_ERROR("Not a single particle??? "
                                 << intersection.size());
                }

                const std::pair<point3, point3>& bot_param =
                    evt_data.m_meas_to_param_map.at(bot_meas);
                const std::pair<point3, point3>& mid_param =
                    evt_data.m_meas_to_param_map.at(mid_meas);
                const std::pair<point3, point3>& top_param =
                    evt_data.m_meas_to_param_map.at(top_meas);

                std::tuple<measurement, measurement, measurement> meas_triple{
                    bot_meas, mid_meas, top_meas};

                if (auto it =
                        seed_to_output_data_point_idx_map.find(meas_triple);
                    it != seed_to_output_data_point_idx_map.end()) {
                    output_data_point& dp = output_data_points.at(it->second);

                    if (dp.found_in_truth) {
                        TRACCC_WARNING(
                            "Attempting to mark seed as found in truth, but it "
                            "already was!");
                    }

                    dp.found_in_truth = true;
                } else {
                    output_data_point new_dp;

                    new_dp.ptc_id = intersection[0].particle_id;
                    new_dp.found_in_truth = true;
                    new_dp.found_in_reco = false;
                    new_dp.bot_x = bot_param.first.at(0);
                    new_dp.bot_y = bot_param.first.at(1);
                    new_dp.bot_z = bot_param.first.at(2);
                    new_dp.mid_x = mid_param.first.at(0);
                    new_dp.mid_y = mid_param.first.at(1);
                    new_dp.mid_z = mid_param.first.at(2);
                    new_dp.top_x = top_param.first.at(0);
                    new_dp.top_y = top_param.first.at(1);
                    new_dp.top_z = top_param.first.at(2);

                    seed_to_output_data_point_idx_map.insert(
                        {meas_triple, output_data_points.size()});
                    output_data_points.push_back(new_dp);
                }

                num_truth_tracks++;
            }

            if (num_untrackable_particles > 0) {
                TRACCC_WARNING(
                    "Found "
                    << num_untrackable_particles
                    << " track candidates with fewer than 3 measurements");
            }

            TRACCC_INFO("The truth seed output has size " << num_truth_tracks);
        }

        {
            traccc::edm::seed_collection::buffer seeds_cuda_buffer;

            traccc::edm::spacepoint_collection::buffer spacepoints_cuda_buffer(
                static_cast<unsigned int>(spacepoints_per_event.size()),
                mr.main);
            async_copy.setup(spacepoints_cuda_buffer)->wait();
            async_copy(vecmem::get_data(spacepoints_per_event),
                       spacepoints_cuda_buffer)
                ->wait();
            traccc::measurement_collection_types::buffer
                measurements_cuda_buffer(
                    static_cast<unsigned int>(measurements_per_event.size()),
                    mr.main);
            async_copy.setup(measurements_cuda_buffer)->wait();
            async_copy(vecmem::get_data(measurements_per_event),
                       measurements_cuda_buffer)
                ->wait();

            // Reconstruct the spacepoints into seeds.
            seeds_cuda_buffer = sa_cuda(spacepoints_cuda_buffer);

            stream.synchronize();

            traccc::edm::seed_collection::device seeds_cuda(seeds_cuda_buffer);

            TRACCC_INFO("The CUDA seed output has size "
                        << seeds_cuda.bottom_index().size());

            if (measurements_per_event.size() != spacepoints_per_event.size()) {
                TRACCC_WARNING(
                    "Number of measurements and spacepoints is not the same; "
                    "results may not be correct.");
            }

            // Do these SoA objects have more easily accessible size methods?
            for (unsigned int i = 0; i < seeds_cuda.bottom_index().size();
                 ++i) {
                const unsigned int bot_idx = seeds_cuda.bottom_index().at(i);
                const unsigned int mid_idx = seeds_cuda.middle_index().at(i);
                const unsigned int top_idx = seeds_cuda.top_index().at(i);

                // NOTE: This assumes that meas[i] matches sp[i].
                const auto& bot_spacepoint = spacepoints_per_event.at(bot_idx);
                const auto& mid_spacepoint = spacepoints_per_event.at(mid_idx);
                const auto& top_spacepoint = spacepoints_per_event.at(top_idx);

                const std::size_t bot_evt_data_meas_id =
                    measurement_to_evt_data_measurement_map.at(
                        measurements_per_event.at(bot_idx));
                const std::size_t mid_evt_data_meas_id =
                    measurement_to_evt_data_measurement_map.at(
                        measurements_per_event.at(mid_idx));
                const std::size_t top_evt_data_meas_id =
                    measurement_to_evt_data_measurement_map.at(
                        measurements_per_event.at(top_idx));

                const measurement& bot_evt_data_meas =
                    evt_data.m_measurement_map.at(bot_evt_data_meas_id);
                const measurement& mid_evt_data_meas =
                    evt_data.m_measurement_map.at(mid_evt_data_meas_id);
                const measurement& top_evt_data_meas =
                    evt_data.m_measurement_map.at(top_evt_data_meas_id);

                const std::map<particle, std::size_t>& bot_ptc_map =
                    evt_data.m_meas_to_ptc_map.at(bot_evt_data_meas);
                const std::map<particle, std::size_t>& mid_ptc_map =
                    evt_data.m_meas_to_ptc_map.at(mid_evt_data_meas);
                const std::map<particle, std::size_t>& top_ptc_map =
                    evt_data.m_meas_to_ptc_map.at(top_evt_data_meas);

                // Calculate the set intersection between the particle maps.
                std::vector<particle> intersection;
                for (const auto& it_bot : bot_ptc_map) {
                    for (const auto& it_mid : mid_ptc_map) {
                        for (const auto& it_top : top_ptc_map) {
                            if (it_bot.first.particle_id ==
                                    it_mid.first.particle_id &&
                                it_mid.first.particle_id ==
                                    it_top.first.particle_id) {
                                intersection.push_back(it_bot.first);
                            }
                        }
                    }
                }

                if (intersection.size() > 1) {
                    TRACCC_WARNING("Track with more than one particle?");
                }

                std::tuple<measurement, measurement, measurement> meas_triple{
                    bot_evt_data_meas, mid_evt_data_meas, top_evt_data_meas};

                if (auto it =
                        seed_to_output_data_point_idx_map.find(meas_triple);
                    it != seed_to_output_data_point_idx_map.end()) {
                    output_data_point& dp = output_data_points.at(it->second);

                    if (dp.found_in_reco) {
                        TRACCC_WARNING(
                            "Attempting to mark seed as found in reco, but it "
                            "already was!");
                    }

                    dp.found_in_reco = true;
                } else {
                    output_data_point new_dp;

                    new_dp.ptc_id = intersection.size() > 0
                                        ? intersection.at(0).particle_id
                                        : 0;
                    new_dp.found_in_truth = false;
                    new_dp.found_in_reco = true;
                    new_dp.bot_x = bot_spacepoint.x();
                    new_dp.bot_y = bot_spacepoint.y();
                    new_dp.bot_z = bot_spacepoint.z();
                    new_dp.mid_x = mid_spacepoint.x();
                    new_dp.mid_y = mid_spacepoint.y();
                    new_dp.mid_z = mid_spacepoint.z();
                    new_dp.top_x = top_spacepoint.x();
                    new_dp.top_y = top_spacepoint.y();
                    new_dp.top_z = top_spacepoint.z();

                    seed_to_output_data_point_idx_map.insert(
                        {meas_triple, output_data_points.size()});
                    output_data_points.push_back(new_dp);
                }
            }
        }

        std::ostringstream filename;
        filename << "event" << std::setfill('0') << std::setw(9) << event
                 << "-seeding-comparison.csv";

        TRACCC_INFO("Writing a total of " << output_data_points.size()
                                          << " datapoints to "
                                          << filename.str());

        std::ofstream file;
        file.open(filename.str());
        file << "particle_id,found_in_truth,found_in_reco,bot_x,bot_y,bot_z,"
                "mid_x,mid_y,mid_z,top_x,top_y,top_z"
             << std::endl;

        for (std::size_t i = 0; i < output_data_points.size(); ++i) {
            const output_data_point& dp = output_data_points.at(i);

            file << dp.ptc_id << "," << dp.found_in_truth << ","
                 << dp.found_in_reco << "," << dp.bot_x << "," << dp.bot_y
                 << "," << dp.bot_z << "," << dp.mid_x << "," << dp.mid_y << ","
                 << dp.mid_z << "," << dp.top_x << "," << dp.top_y << ","
                 << dp.top_z << std::endl;
        }

        file.close();
    }

    return 0;
}

int main(int argc, char* argv[]) {
    std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
        "TracccExampleTruthFindingCuda", traccc::Logging::Level::INFO);

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_seeding seeding_opts;
    traccc::opts::program_options program_opts{
        "Seeding Comparison Using CUDA",
        {detector_opts, input_opts, seeding_opts},
        argc,
        argv,
        logger->cloneWithSuffix("Options")};

    // Run the application.
    return seq_run(input_opts, detector_opts, seeding_opts, logger->clone());
}
