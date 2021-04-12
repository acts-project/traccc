/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "cpu/include/algorithms/component_connection.hpp"
#include "cpu/include/algorithms/measurement_creation.hpp"
#include "cpu/include/algorithms/spacepoint_formation.hpp"
#include "cuda/include/algorithms/spacepoint_formation/component_connection_kernels.cuh"
#include "cuda/include/algorithms/spacepoint_formation/spacepoint_formation_kernels.cuh"
#include "geometry/pixel_segmentation.hpp"
#include "csv/csv_io.hpp"
#include <iostream>
#include <chrono>

int seq_run(const std::string& detector_file, const std::string& cells_dir, unsigned int events)
{
    auto env_d_d = std::getenv("TRACCC_TEST_DATA_DIR");
    if (env_d_d == nullptr)
    {
        throw std::ios_base::failure("Test data directory not found. Please set TRACCC_TEST_DATA_DIR.");
    }
    auto data_directory = std::string(env_d_d) + std::string("/");

    // Read the surface transforms
    std::string io_detector_file = data_directory + detector_file;
    traccc::surface_reader sreader(io_detector_file, {"geometry_id", "cx", "cy", "cz", "rot_xu", "rot_xv", "rot_xw", "rot_zu", "rot_zv", "rot_zw"});
    auto surface_transforms = traccc::read_surfaces(sreader);

    // Algorithms
    traccc::component_connection cc;
    traccc::measurement_creation mt;
    traccc::spacepoint_formation sp;

    // Output stats
    uint64_t n_cells = 0;
    uint64_t m_modules = 0;
    uint64_t n_clusters = 0;
    uint64_t n_measurements = 0;
    uint64_t n_space_points = 0;

    // Time Elapsed
    double cpu_time = 0;
    double cuda_time = 0;
    double read_time = 0;
    double copy_time = 0;
    
    // Loop over events
    for (unsigned int event = 0; event < events; ++event){
      
        // Read the cells from the relevant event file
        std::string event_string = "000000000";
        std::string event_number = std::to_string(event);
        event_string.replace(event_string.size()-event_number.size(), event_number.size(), event_number);

        std::string io_cells_file = data_directory+cells_dir+std::string("/event")+event_string+std::string("-cells.csv");
        traccc::cell_reader creader(io_cells_file, {"geometry_id", "hit_id", "cannel0", "channel1", "activation", "time"});

	//------------------
	// CPU part
	//------------------


	auto start_read = std::chrono::system_clock::now();
	
	traccc::cell_container cells_per_event = traccc::read_cells(creader, surface_transforms);

	auto end_read = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsec_read = end_read - start_read;
	read_time += elapsec_read.count();
	
        m_modules += cells_per_event.size();

	auto start_cpu = std::chrono::system_clock::now();
	
        // Output containers
        traccc::measurement_container measurements_per_event;
        traccc::spacepoint_container spacepoints_per_event;
        measurements_per_event.reserve(cells_per_event.size());
        spacepoints_per_event.reserve(cells_per_event.size());
       
        for (auto &cells_per_module : cells_per_event)
        {
            // The algorithmic code part: start
            traccc::cluster_collection clusters_per_module =  cc(cells_per_module);  
            clusters_per_module.position_from_cell = traccc::pixel_segmentation{-8.425, -36.025, 0.05, 0.05};
            traccc::measurement_collection measurements_per_module = mt(clusters_per_module);
            traccc::spacepoint_collection spacepoints_per_module = sp(measurements_per_module);
            // The algorithmnic code part: end            
            n_cells += cells_per_module.items.size();
            n_clusters += clusters_per_module.items.size();
            n_measurements += measurements_per_module.items.size();
            n_space_points += spacepoints_per_module.items.size();

            measurements_per_event.push_back(std::move(measurements_per_module));
            spacepoints_per_event.push_back(std::move(spacepoints_per_module));
        }
	auto end_cpu = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsec_cpu = end_cpu - start_cpu;
	cpu_time += elapsec_cpu.count();
	
	//------------------
	// CUDA part
	//------------------

	auto start_copy = std::chrono::system_clock::now();
	
	traccc::cell_container_cuda cells_cuda;
	// Copy cpu cells to vecmem
	for (auto cell_module: cells_per_event){
	    vecmem::vector<traccc::cell> cell_copy(cell_module.items.begin(),cell_module.items.end());
	    cells_cuda.items.push_back(cell_copy);
	    cells_cuda.modcfg.push_back(cell_module.modcfg);
	}

	auto end_copy = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsec_copy = end_copy - start_copy;
	copy_time += elapsec_copy.count();
	
	auto start_cuda = std::chrono::system_clock::now();
	
	traccc::label_container_cuda labels_cuda(cells_cuda);

	// Run sparseCCL
	traccc::sparse_ccl_cuda(cells_cuda, labels_cuda);
	
	traccc::measurement_container_cuda measurements_cuda(cells_cuda,labels_cuda);
	traccc::spacepoint_container_cuda spacepoints_cuda(cells_cuda,labels_cuda);
	
	// Run spacepoint formation	
	traccc::sp_formation_cuda(cells_cuda,
				  labels_cuda,
				  measurements_cuda,
				  spacepoints_cuda);
	
	auto end_cuda = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsec_cuda = end_cuda - start_cuda;
	cuda_time += elapsec_cuda.count();
	
	// measurement validation
	for (int i=0; i<measurements_cuda.items.size(); i++){	
	    auto ms_obj = measurements_cuda.items[i];
	    for (int j=0; j<ms_obj.size(); j++){
		auto ms_cpu = (measurements_per_event[i].items)[j];
		
		if ( (ms_obj[j].local[0] != ms_cpu.local[0]) ||
		     (ms_obj[j].local[1] != ms_cpu.local[1]) ||
		     (ms_obj[j].local[2] != ms_cpu.local[2]) ){
		    std::cout << "there is a problem in measurement creation" << std::endl;
	    }
	  }
	}
	
	// spacepoint validation
	for (int i=0; i<spacepoints_cuda.items.size(); i++){
	  auto sp_obj = spacepoints_cuda.items[i];
	  for (int j=0; j<sp_obj.size(); j++){
	    auto sp_cpu = (spacepoints_per_event[i].items)[j];
	    if ( (sp_obj[j].global[0] != sp_cpu.global[0]) ||
		 (sp_obj[j].global[1] != sp_cpu.global[1]) ||
		 (sp_obj[j].global[2] != sp_cpu.global[2]) ){
	      std::cout << "there is a problem in space formation..." << std::endl;
	    }
	  }
	}
	
        traccc::measurement_writer mwriter{std::string("event")+event_number+"-measurements.csv"};
        for (const auto& measurements_per_module : measurements_per_event){
            auto module = measurements_per_module.modcfg.module;
            for (const auto& measurement : measurements_per_module.items){
                const auto& local = measurement.local;
                mwriter.append({ module, local[0], local[1], 0., 0.});
            }
        }

        traccc::spacepoint_writer spwriter{std::string("event")+event_number+"-spacepoints.csv"};
        for (const auto& spacepoint_per_module : spacepoints_per_event){
            auto module = spacepoint_per_module.module;
            for (const auto& spacepoint : spacepoint_per_module.items){
                const auto& pos = spacepoint.global;
                spwriter.append({ module, pos[0], pos[1], pos[2], 0., 0., 0.});
            }
        }
	
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells from " << m_modules << " modules" << std::endl;
    std::cout << "- created " << n_clusters << " clusters. " << std::endl;
    std::cout << "- created " << n_measurements << " measurements. " << std::endl;
    std::cout << "- created " << n_space_points << " space points. " << std::endl;

    std::cout << "==> Elapsed time ... " << std::endl;
    std::cout << "- cpu data reading time: " << read_time << std::endl;
    std::cout << "- cpu-to-gpu cell-data copying time: " << copy_time << std::endl;   
    std::cout << "- cpu time: " << cpu_time << std::endl;
    std::cout << "- cuda time: " << cuda_time << std::endl;
    
    return 0;
}

// The main routine
//
int main(int argc, char *argv[])
{
    if (argc < 4){
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << "./seq_example <detector_file> <cell_directory> <events>" << std::endl;
        return -1;
    }

    auto detector_file = std::string(argv[1]);
    auto cell_directory = std::string(argv[2]);
    auto events = std::atoi(argv[3]);

    std::cout << "Running ./seq_exammple " << detector_file << " " << cell_directory << " " << events << std::endl;
    return seq_run(detector_file, cell_directory, events);
}
