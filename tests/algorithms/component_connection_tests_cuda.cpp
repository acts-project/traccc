/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include "vecmem/containers/array.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/utils/cuda/copy.hpp"
#include "clusterization/component_connection_kernels.cuh"

#include <gtest/gtest.h>

// This defines the local frame test suite
TEST(algorithms, component_connection_cuda){

  std::vector<traccc::cell> cell_items = 
    { {1, 0, 1., 0. }, 
      {8, 4, 2., 0.}, 
      {10, 4, 3., 0.}, 
      {9, 5, 4., 0.}, 
      {10, 5, 5., 0}, 
      {12, 12, 6, 0}, 
      {3, 13, 7, 0}, 
      {11, 13, 8, 0}, 
      {4, 14, 9, 0 } };

  vecmem::cuda::managed_memory_resource managed_resource;

  vecmem::jagged_vector< traccc::cell > cell_per_event(&managed_resource);
  vecmem::vector< traccc::cell > cell_per_module(&managed_resource);

  // move cell_items to cell_per_module
  for (auto i=0; i<cell_items.size(); i++){
    traccc::cell aCell;
    aCell.channel0 = std::move(cell_items[i].channel0);
    aCell.channel1 = std::move(cell_items[i].channel1);
    aCell.activation = std::move(cell_items[i].activation);
    aCell.time = std::move(cell_items[i].time);
    cell_per_module.push_back(aCell);
  }

  vecmem::vector< unsigned int > label_per_module(cell_per_module.size(),0,&managed_resource);
  vecmem::jagged_vector< unsigned int > label_per_event(&managed_resource);
  
  // fill jagged_vector
  cell_per_event.push_back(cell_per_module);
  label_per_event.push_back(label_per_module);

  vecmem::vector< unsigned int> num_labels(cell_per_event.size(), 0 , &managed_resource);

  // run sparse_ccl
  vecmem::data::jagged_vector_data< traccc::cell> cell_data(cell_per_event,&managed_resource);
  vecmem::data::jagged_vector_data< unsigned int > label_data(label_per_event, &managed_resource);
  
  traccc::sparse_ccl_cuda(cell_data,
			  label_data,
			  vecmem::get_data( num_labels ));

  // run spacepoint formation
  vecmem::jagged_vector< traccc::measurement > ms_per_event(&managed_resource);
  vecmem::vector< traccc::measurement > ms_per_module(num_labels[0],&managed_resource);
  ms_per_event.push_back(ms_per_module);
  vecmem::data::jagged_vector_data< traccc::measurement > ms_data(ms_per_event, &managed_resource);
  
  vecmem::jagged_vector< traccc::spacepoint > sp_per_event(&managed_resource);
  vecmem::vector< traccc::spacepoint > sp_per_module(num_labels[0],&managed_resource);
  sp_per_event.push_back(sp_per_module);
  vecmem::data::jagged_vector_data< traccc::spacepoint > sp_data(sp_per_event, &managed_resource);
  
  traccc::sp_formation_cuda(cell_data,
			    label_data,
			    vecmem::get_data( num_labels ),
			    ms_data,
			    sp_data);
  
  //std::cout << num_labels[0] << std::endl;
}

// Google Test can be run manually from the main() function
// or, it can be linked to the gtest_main library for an already
// set-up main() function primed to accept Google Test test cases.
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
