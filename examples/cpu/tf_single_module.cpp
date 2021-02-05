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
#include "algorithms/component_connection.hpp"
#include "algorithms/measurement_creation.hpp"
#include "algorithms/spacepoint_formation.hpp"


#include <taskflow/taskflow.hpp>  

int main(){
  
  tf::Executor executor;
  tf::Taskflow taskflow;

  /// Following [DOI: 10.1109/DASIP48288.2019.9049184]
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


  traccc::cell_collection cells;
  cells.items = cell_items;
  cells.module_id = 0;

  traccc::cluster_collection clusters;
  clusters.position_from_cell = traccc::pixel_segmentation{0.,0.,1.,1.};

  traccc::measurement_collection measurements;
  measurements.placement = traccc::transform3{};
  
  traccc::spacepoint_collection spacepoints;

  traccc::component_connection cc;
  traccc::measurement_creation mt;
  traccc::spacepoint_formation sp; 

  auto [ ccl_task, meas_task, sp_task ] 
    = taskflow.emplace( [&](){ cc(cells, clusters); }, 
                        [&](){ mt(clusters, measurements); }, 
                        [&](){ sp(measurements, spacepoints);} );
                                      
  ccl_task.precede(meas_task);  
  meas_task.precede(sp_task);

  executor.run(taskflow).wait(); 

  return 0;
}