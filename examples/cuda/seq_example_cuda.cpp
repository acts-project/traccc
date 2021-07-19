/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <iostream>
#include <vecmem/memory/host_memory_resource.hpp>

#include "csv/csv_io.hpp"
#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/internal_spacepoint.hpp"
#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"
#include "geometry/pixel_segmentation.hpp"

// clusterization (cpu)
#include "clusterization/component_connection.hpp"
#include "clusterization/measurement_creation.hpp"
#include "clusterization/spacepoint_formation.hpp"

// seeding (cpu)
#include "seeding/seed_finding.hpp"
#include "seeding/spacepoint_grouping.hpp"
// seeding (cuda)
#include "cuda/seeding/seed_finding.hpp"

// io
#include "csv/csv_io.hpp"

// vecmem
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

// std
#include <chrono>
#include <iomanip>

// custom
#include "tml_stats_config.hpp"


int seq_run(const std::string& detector_file, const std::string& cells_dir,
	    unsigned int skip_events, unsigned int events, bool skip_cpu,
            bool skip_write) {
    // Read the surface transforms
    std::string io_detector_file = detector_file;
    traccc::surface_reader sreader(
        io_detector_file, {"geometry_id", "cx", "cy", "cz", "rot_xu", "rot_xv",
                           "rot_xw", "rot_zu", "rot_zv", "rot_zw"});
    auto surface_transforms = traccc::read_surfaces(sreader);

    // Algorithms
    traccc::component_connection cc;
    traccc::measurement_creation mt;
    traccc::spacepoint_formation sp;

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    uint64_t n_clusters = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_internal_spacepoints = 0;
    uint64_t n_doublets = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_cuda = 0;

    // Elapsed time
    float wall_time(0);

    float file_reading_cpu(0);
    float clusterization_cpu(0);	
    float measurement_creation_cpu(0);
    float spacepoint_formation_cpu(0);
    float binning_cpu(0);
    float seeding_cpu(0);

    float clusterization_cuda(0);	
    float measurement_creation_cuda(0);
    float spacepoint_formation_cuda(0);
    float binning_cuda(0);
    float seeding_cuda(0);
    
    // Memory resource used by the EDM.
    vecmem::host_memory_resource resource;

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    
    /*------------------------------
      Spacepoint grid configuration
      ------------------------------*/

    // Seed finder config
    traccc::seedfinder_config config;
    // silicon detector max
    config.rMax = 160.;
    config.deltaRMin = 5.;
    config.deltaRMax = 160.;
    config.collisionRegionMin = -250.;
    config.collisionRegionMax = 250.;
    //config.zMin = -2800.; // this value introduces redundant bins without any
    //config.zMax = 2800.;
    config.zMin = -1186.;
    config.zMax = 1186.;
    config.maxSeedsPerSpM = 5;
    // 2.7 eta
    config.cotThetaMax = 7.40627;
    config.sigmaScattering = 1.00000;

    config.minPt = 500.;
    config.bFieldInZ = 0.00199724;

    config.beamPos = {-.5, -.5};
    config.impactMax = 10.;

    config.highland = 13.6 * std::sqrt(config.radLengthPerSeed) *
                      (1 + 0.038 * std::log(config.radLengthPerSeed));
    float maxScatteringAngle = config.highland / config.minPt;
    config.maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;
    // helix radius in homogeneous magnetic field. Units are Kilotesla, MeV and
    // millimeter
    // TODO: change using ACTS units
    config.pTPerHelixRadius = 300. * config.bFieldInZ;
    config.minHelixDiameter2 =
        std::pow(config.minPt * 2 / config.pTPerHelixRadius, 2);
    config.pT2perRadius =
        std::pow(config.highland / config.pTPerHelixRadius, 2);

    // setup spacepoint grid config
    traccc::spacepoint_grid_config grid_config;
    grid_config.bFieldInZ = config.bFieldInZ;
    grid_config.minPt = config.minPt;
    grid_config.rMax = config.rMax;
    grid_config.zMax = config.zMax;
    grid_config.zMin = config.zMin;
    grid_config.deltaRMax = config.deltaRMax;
    grid_config.cotThetaMax = config.cotThetaMax;

    traccc::spacepoint_grouping sg(config, grid_config);
    traccc::seed_finding sf(config);
    
    traccc::cuda::tml_stats_config tml_cfg;
    traccc::cuda::seed_finding sf_cuda(config, sg.get_spgrid(), &tml_cfg,
                                       &mng_mr);    

    /*time*/ auto start_wall_time = std::chrono::system_clock::now();
    
    // Loop over events
    for (unsigned int event = skip_events; event < skip_events + events;
         ++event) {

        // Read the cells from the relevant event file
	/*time*/ auto start_file_reading_cpu = std::chrono::system_clock::now();
	
        std::string event_string = "000000000";
        std::string event_number = std::to_string(event);
        event_string.replace(event_string.size() - event_number.size(),
                             event_number.size(), event_number);
	
	
        std::string io_cells_file = cells_dir +
                                    std::string("/event") + event_string +
                                    std::string("-cells.csv");
        traccc::cell_reader creader(
            io_cells_file, {"geometry_id", "hit_id", "cannel0", "channel1",
                            "activation", "time"});
        traccc::host_cell_container cells_per_event =
            traccc::read_cells(creader, resource, surface_transforms);
        n_modules += cells_per_event.headers.size();
	
	
        /*time*/ auto end_file_reading_cpu = std::chrono::system_clock::now();
        /*time*/ std::chrono::duration<double> time_file_reading_cpu =
            end_file_reading_cpu - start_file_reading_cpu;
        /*time*/ file_reading_cpu += time_file_reading_cpu.count();
	
        /*-----------------------------
          spacepoint formation - cpu
          -----------------------------*/
	
        traccc::host_measurement_container measurements_per_event;
        traccc::host_spacepoint_container spacepoints_per_event;
        measurements_per_event.headers.reserve(cells_per_event.headers.size());
        measurements_per_event.items.reserve(cells_per_event.headers.size());
        spacepoints_per_event.headers.reserve(cells_per_event.headers.size());
        spacepoints_per_event.items.reserve(cells_per_event.headers.size());

        for (std::size_t i = 0; i < cells_per_event.items.size(); ++i) {
            auto& module = cells_per_event.headers[i];
            module.pixel =
                traccc::pixel_segmentation{-8.425, -36.025, 0.05, 0.05};

	    /*time*/ auto start_clusterization_cpu = std::chrono::system_clock::now();
	    
            // The algorithmic code part: start
            traccc::cluster_collection clusters_per_module =
                cc({cells_per_event.items[i], cells_per_event.headers[i]});
            clusters_per_module.position_from_cell = module.pixel;

	    /*time*/ auto end_clusterization_cpu = std::chrono::system_clock::now();
	    /*time*/ std::chrono::duration<double> time_clusterization_cpu =
		end_clusterization_cpu - start_clusterization_cpu;
	    /*time*/ clusterization_cpu += time_clusterization_cpu.count();
	    
	    /*time*/ auto start_measurement_creation_cpu = std::chrono::system_clock::now();
	    
            traccc::host_measurement_collection measurements_per_module =
                mt({clusters_per_module, module});

	    /*time*/ auto end_measurement_creation_cpu = std::chrono::system_clock::now();
	    /*time*/ std::chrono::duration<double> time_measurement_creation_cpu =
		end_measurement_creation_cpu - start_measurement_creation_cpu;
	    /*time*/ measurement_creation_cpu += time_measurement_creation_cpu.count();
	    
	    /*time*/ auto start_spacepoint_formation_cpu = std::chrono::system_clock::now();
	    
            traccc::host_spacepoint_collection spacepoints_per_module =
                sp({module, measurements_per_module});

	    /*time*/ auto end_spacepoint_formation_cpu = std::chrono::system_clock::now();
	    /*time*/ std::chrono::duration<double> time_spacepoint_formation_cpu =
		end_spacepoint_formation_cpu - start_spacepoint_formation_cpu;
	    /*time*/ spacepoint_formation_cpu += time_spacepoint_formation_cpu.count();
            // The algorithmnic code part: end
	    
            n_cells += cells_per_event.items[i].size();
            n_clusters += clusters_per_module.items.size();
            n_measurements += measurements_per_module.size();
            n_spacepoints += spacepoints_per_module.size();

            measurements_per_event.items.push_back(
                std::move(measurements_per_module));
            measurements_per_event.headers.push_back(module);

            spacepoints_per_event.items.push_back(
                std::move(spacepoints_per_module));
            spacepoints_per_event.headers.push_back(module.module);
        }
		
        /*-------------------
          spacepoint binning
          -------------------*/

        // create internal spacepoints grouped in bins
        /*time*/ auto start_binning_cpu = std::chrono::system_clock::now();

        auto internal_sp_per_event = sg(spacepoints_per_event, &mng_mr);

        /*time*/ auto end_binning_cpu = std::chrono::system_clock::now();
        /*time*/ std::chrono::duration<double> time_binning_cpu =
            end_binning_cpu - start_binning_cpu;

        /*time*/ binning_cpu += time_binning_cpu.count();
	
        /*-----------------------
          seed finding -- cpu
          -----------------------*/

	traccc::host_seed_container seeds;

        if (skip_cpu == false) {
	    /*time*/ auto start_seeding_cpu = std::chrono::system_clock::now();
	    
            seeds = sf(internal_sp_per_event);
            //n_seeds += seeds.size();
	    n_seeds += seeds.headers[0];

            /*time*/ auto end_seeding_cpu = std::chrono::system_clock::now();
            /*time*/ std::chrono::duration<double> time_seeding_cpu =
                end_seeding_cpu - start_seeding_cpu;
            /*time*/ seeding_cpu += time_seeding_cpu.count();

            for (size_t i = 0; i < internal_sp_per_event.headers.size(); ++i) {
                n_internal_spacepoints += internal_sp_per_event.items[i].size();
            }
        }
	
        /*-----------------------
          seed finding -- cuda
          -----------------------*/

        /*time*/ auto start_seeding_cuda = std::chrono::system_clock::now();
        auto seeds_cuda = sf_cuda(internal_sp_per_event);
        n_seeds_cuda += seeds_cuda.headers[0];

        /*time*/ auto end_seeding_cuda = std::chrono::system_clock::now();
        /*time*/ std::chrono::duration<double> time_seeding_cuda =
            end_seeding_cuda - start_seeding_cuda;
        /*time*/ seeding_cuda += time_seeding_cuda.count();
	
        /*----------------------------------
          compare seeds from cpu and cuda
          ----------------------------------*/

        if (!skip_cpu) {
            int n_match = 0;
	    for (auto seed : seeds.items[0]) {
                if (std::find(
                        seeds_cuda.items[0].begin(),
                        seeds_cuda.items[0].begin() + seeds_cuda.headers[0],
                        seed) !=
                    seeds_cuda.items[0].begin() + seeds_cuda.headers[0]) {
                    n_match++;
                }
            }
	    float matching_rate = float(n_match) / seeds.headers[0];
            std::cout << "event " << std::to_string(skip_events + event)
                      << " seed matching rate: " << matching_rate << std::endl;
        }
	
        /*------------
             Writer
          ------------*/

	if (!skip_write){
	    /*
	    traccc::measurement_writer mwriter{std::string("event") + event_number +
					       "-measurements.csv"};
	    for (size_t i = 0; i < measurements_per_event.items.size(); ++i) {
		auto measurements_per_module = measurements_per_event.items[i];
		auto module = measurements_per_event.headers[i];
		for (const auto& measurement : measurements_per_module) {
		    const auto& local = measurement.local;
		    mwriter.append({module.module, local[0], local[1], 0., 0.});
		}
	    }
	    */
	    traccc::spacepoint_writer spwriter{std::string("event") + event_number +
		    "-spacepoints.csv"};
	    for (size_t i = 0; i < spacepoints_per_event.items.size(); ++i) {
		auto spacepoints_per_module = spacepoints_per_event.items[i];
		auto module = spacepoints_per_event.headers[i];

		for (const auto& spacepoint : spacepoints_per_module) {
		    const auto& pos = spacepoint.global;
		    spwriter.append({module, pos[0], pos[1], pos[2], 0., 0., 0.});
		}
	    }
	    
	    traccc::internal_spacepoint_writer internal_spwriter{
		std::string("event") + event_number + "-internal_spacepoints.csv"};
	    for (size_t i = 0; i < internal_sp_per_event.items.size(); ++i) {
		auto internal_sp_per_bin = internal_sp_per_event.items[i];
		auto bin = internal_sp_per_event.headers[i].global_index;
		
		for (const auto& internal_sp : internal_sp_per_bin) {
		    const auto& x = internal_sp.m_x;
		    const auto& y = internal_sp.m_y;
		    const auto& z = internal_sp.m_z;
		    const auto& varR = internal_sp.m_varianceR;
		    const auto& varZ = internal_sp.m_varianceZ;
		    internal_spwriter.append({bin, x, y, z, varR, varZ});
		}
	    }
	}
    }

    /*time*/ auto end_wall_time = std::chrono::system_clock::now();
    /*time*/ std::chrono::duration<double> time_wall_time =
        end_wall_time - start_wall_time;

    /*time*/ wall_time += time_wall_time.count();

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints from "
              << n_modules << " modules" << std::endl;
    std::cout << "- created        " << n_cells
              << " cells           " << std::endl;
    std::cout << "- created        " << n_clusters
              << " clusters        " << std::endl;        
    std::cout << "- created        " << n_measurements
              << " meaurements     " << std::endl;
    std::cout << "- created        " << n_spacepoints
              << " spacepoints     " << std::endl;
    std::cout << "- created        " << n_internal_spacepoints
              << " internal spacepoints" << std::endl;

    std::cout << "- created (cpu)  " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (cuda) " << n_seeds_cuda << " seeds" << std::endl;
    std::cout << "==> Elpased time ... " << std::endl;
    std::cout << "wall time           " << std::setw(10) << std::left
              << wall_time << std::endl;
    std::cout << "file reading (cpu)       " << std::setw(10) << std::left
              << file_reading_cpu << std::endl;
    std::cout << "clusterization_time (cpu)" << std::setw(10) << std::left
              << clusterization_cpu << std::endl;    
    std::cout << "ms_creation_time (cpu)   " << std::setw(10) << std::left
              << measurement_creation_cpu << std::endl;    
    std::cout << "sp_formation_time (cpu)  " << std::setw(10) << std::left
              << spacepoint_formation_cpu << std::endl;    
    std::cout << "binning_time (cpu)       " << std::setw(10) << std::left
              << binning_cpu << std::endl;
    std::cout << "seeding_time (cpu)       " << std::setw(10) << std::left
              << seeding_cpu << std::endl;
    std::cout << "seeding_time (cuda)      " << std::setw(10) << std::left
              << seeding_cuda << std::endl;

    return 0;    
}

// The main routine
//
int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << "./seq_example <detector_file> <hit_directory> "
                     "<skip_events> <events> <skip_cpu> <skip_write>"
                  << std::endl;
        return -1;
    }

    auto detector_file = std::string(argv[1]);
    auto hit_directory = std::string(argv[2]);
    auto skip_events = std::atoi(argv[3]);
    auto events = std::atoi(argv[4]);
    bool skip_cpu = std::atoi(argv[5]);
    bool skip_write = std::atoi(argv[6]);

    std::cout << "Running ./seq_example " << detector_file << " "
              << hit_directory << " " << skip_events << " " << events << " "
              << skip_cpu << " " << skip_write << std::endl;
    return seq_run(detector_file, hit_directory, skip_events, events, skip_cpu,
                   skip_write);
}
