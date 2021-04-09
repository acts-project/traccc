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
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/data/jagged_vector_data.hpp"
#include "clusterization/component_connection_kernels.cuh"
#include "white_paper_kernels.cuh"

#include <iostream>
#include <chrono>
#include <memory_resource>


int main(int argc, char *argv[]){

    vecmem::cuda::managed_memory_resource mr;
    traccc::cell_event cell_evt;    

    traccc::cell test_cell1{1,2,3,4};
    traccc::cell test_cell2{5,6,7,8};
    vecmem::vector<traccc::cell> cell_vec({test_cell1, test_cell2});
    /*
    cell_evt.items = vecmem::jagged_vector<traccc::cell>({cell_vec}, &mr);
    vecmem::data::jagged_vector_data< traccc::cell > cells(cell_evt.items, &mr);
    cell_test(cells);
    */

    //vecmem::vector< traccc::module_config >config(&mr);
    vecmem::jagged_vector<traccc::cell>cell_jag({cell_vec},&mr);
    vecmem::data::jagged_vector_data< traccc::cell > cells(cell_jag, &mr);

    cell_test(cells);
    
    ////

    int a = 1;
    int b = 2;
    vecmem::vector< int > int_vec({a,b},&mr);
    vecmem::jagged_vector< int > int_jag({int_vec}, &mr);
    vecmem::data::jagged_vector_data< int > int_data(int_jag, &mr);
    int_test(int_data);
    
    return 0;
}


