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
#include "algorithms/clustering.hpp"
#include "algorithms/measurement_creation.hpp"
#include "algorithms/spacepoint_formation.hpp"

#include <taskflow/taskflow.hpp>  

int main(){
  
  tf::Executor executor;
  tf::Taskflow taskflow;

  traccc::cell_collection cells;
  traccc::cluster_collection clusters;
  traccc::measurement_collection measurements;
  traccc::spacepoint_collection spacepoints;

  traccc::clustering cl;
  traccc::measurement_creation mt;
  traccc::spacepoint_formation sp; 

  auto [ clus_task, meas_task, sp_task ] 
    = taskflow.emplace( [&](){ cl(cells, clusters); }, 
                        [&](){ mt(clusters, measurements); }, 
                        [&](){sp(measurements, spacepoints);} );
                                      
  clus_task.precede(meas_task);  
  sp_task.precede(meas_task);

  executor.run(taskflow).wait(); 

  return 0;
}