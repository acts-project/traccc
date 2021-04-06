/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"
#include "geometry/pixel_segmentation.hpp"
#include "component_connection.hpp"
#include "measurement_creation.hpp"
#include "spacepoint_formation.hpp"
#include "csv/csv_io.hpp"


#include "vecmem/containers/array.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/utils/cuda/copy.hpp"
#include "clusterization/component_connection_kernels.cuh"

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

	auto start_cpu = std::chrono::system_clock::now();
	
	traccc::cell_container cells_per_event = traccc::read_cells(creader, surface_transforms);
        m_modules += cells_per_event.size();

        // Output containers
        traccc::measurement_container measurements_per_event;
        traccc::spacepoint_container spacepoints_per_event;
        measurements_per_event.reserve(cells_per_event.size());
        spacepoints_per_event.reserve(cells_per_event.size());

	std::vector< std::vector< unsigned int > > label_per_event; // test
       
        for (auto &cells_per_module : cells_per_event)
        {
            // The algorithmic code part: start
	  auto test =  traccc::detail::sparse_ccl(cells_per_module.items);
	  label_per_event.push_back(std::get<1>(test));
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

	auto start_cuda = std::chrono::system_clock::now();
	
        traccc::cell_reader creader_cuda(io_cells_file, {"geometry_id", "hit_id", "cannel0", "channel1", "activation", "time"});       
	vecmem::cuda::managed_memory_resource managed_resource;
	
	vecmem::jagged_vector< traccc::cell > vm_cells_per_event(&managed_resource);
	vecmem::vector< std::tuple< traccc::geometry_id, traccc::transform3 > > vm_geoInfo_per_event(&managed_resource);
	
	traccc::read_cells_vecmem(vm_cells_per_event,
				  vm_geoInfo_per_event,
				  creader_cuda,
				  surface_transforms);

	// prepare label data
	vecmem::jagged_vector< unsigned int > vm_label_per_event(&managed_resource);
	for (auto& cells: vm_cells_per_event){
	  vm_label_per_event.push_back(vecmem::vector< unsigned int>(cells.size(),0,&managed_resource));
	}	
	vecmem::vector< unsigned int> num_clusters_per_event(vm_cells_per_event.size(), 0 , &managed_resource);

	// run sparse_ccl
	vecmem::data::jagged_vector_data< traccc::cell> cell_data(vm_cells_per_event,&managed_resource);
	vecmem::data::jagged_vector_data< unsigned int > label_data(vm_label_per_event, &managed_resource);
	
	traccc::sparse_ccl_cuda(cell_data,
				label_data,
				vecmem::get_data( num_clusters_per_event ));
	
	
	
	// run space formation
	vecmem::jagged_vector< traccc::measurement > vm_ms_per_event(&managed_resource);
  	vecmem::jagged_vector< traccc::spacepoint > vm_sp_per_event(&managed_resource);

	for(auto& nclusters: num_clusters_per_event){
	  vm_ms_per_event.push_back(vecmem::vector< traccc::measurement >(nclusters));
	  vm_sp_per_event.push_back(vecmem::vector< traccc::spacepoint >(nclusters));
	}
	
	vecmem::data::jagged_vector_data< traccc::measurement > ms_data(vm_ms_per_event, &managed_resource);
	vecmem::data::jagged_vector_data< traccc::spacepoint > sp_data(vm_sp_per_event, &managed_resource);
	
	traccc::sp_formation_cuda(cell_data,
				  label_data,
				  vecmem::get_data( num_clusters_per_event ),
				  vecmem::get_data( vm_geoInfo_per_event ),
				  ms_data,
				  sp_data);	

	auto end_cuda = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsec_cuda = end_cuda - start_cuda;
	cuda_time += elapsec_cuda.count();
	  
	// read test
	for (int i=0; i<vm_geoInfo_per_event.size(); i++){
	  if ((std::get<0>(vm_geoInfo_per_event.at(i))!=cells_per_event[i].module)){
	    std::cout << "there is a problem in reading module ID..." << std::endl;
	  }
	  auto vm_placement = std::get<1>(vm_geoInfo_per_event.at(i));
	  auto cpu_placement = cells_per_event[i].placement;
	  if (vm_placement._data != cpu_placement._data){
	    std::cout << "there is a problem in reading placement..." << std::endl;
	  }
	}
	
	// label index test
	for (int i=0; i<vm_label_per_event.size(); i++){
	  auto obj = vm_label_per_event[i];
	  for (int j=0; j<obj.size(); j++){
	    if (obj[j] != label_per_event[i][j] ||
		obj.size() != label_per_event[i].size() ||
		vm_label_per_event.size() != label_per_event.size()){
	      std::cout << "there is a problem in sparse ccl..." << std::endl;
	    }
	  }
	}

	// ms test
	for (int i=0; i<vm_ms_per_event.size(); i++){	
	  auto ms_obj = vm_ms_per_event[i];
	  for (int j=0; j<ms_obj.size(); j++){
	    auto ms_cpu = (measurements_per_event[i].items)[j];

	    if ( (ms_obj[j].local[0] != ms_cpu.local[0]) ||
		 (ms_obj[j].local[1] != ms_cpu.local[1]) ||
		 (ms_obj[j].local[2] != ms_cpu.local[2]) ){
	      std::cout << "there is a problem in measurement creation" << std::endl;
	    }
	  }
	}
	// sp test
	
	for (int i=0; i<vm_sp_per_event.size(); i++){
	  auto sp_obj = vm_sp_per_event[i];
	  for (int j=0; j<sp_obj.size(); j++){
	    auto sp_cpu = (spacepoints_per_event[i].items)[j];
	    if ( (sp_obj[j].global[0] != sp_cpu.global[0]) ||
		 (sp_obj[j].global[1] != sp_cpu.global[1]) ||
		 (sp_obj[j].global[2] != sp_cpu.global[2]) ){
	      std::cout << "there is a problem in space formation..." << std::endl;
	    }
	    /*
	    printf("%f %f %f %f %f %f \n",
		   sp_obj[j].global[0], sp_obj[j].global[1], sp_obj[j].global[2],
		   sp_cpu.global[0], sp_cpu.global[1], sp_cpu.global[2]);      
	    */
	  }
	}
	
        traccc::measurement_writer mwriter{std::string("event")+event_number+"-measurements.csv"};
        for (const auto& measurements_per_module : measurements_per_event){
            auto module = measurements_per_module.module;
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

    std::cout << "CPU Time: " << cpu_time << "  CUDA Time: " << cuda_time << std::endl;
    
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
