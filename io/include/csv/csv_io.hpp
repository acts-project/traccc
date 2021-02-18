/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "definitions/algebra.hpp"
#include "definitions/primitives.hpp"

#include <dfe/dfe_namedtuple.hpp>
#include <dfe/dfe_io_dsv.hpp>

#include <fstream>
#include <climits>
#include <map>

namespace traccc {

  struct csv_cell {
    
    uint64_t geometry_id = 0;
    uint64_t hit_id = 0;
    channel_id channel0 = 0;
    channel_id channel1 = 0;
    scalar timestamp = 0.;
    scalar value = 0.;

    // geometry_id,hit_id,channel0,channel1,timestamp,value
    DFE_NAMEDTUPLE(csv_cell, geometry_id, hit_id, channel0, channel1, timestamp, value);

  };

  using cell_reader = dfe::NamedTupleCsvReader<csv_cell>;

  struct csv_measurement {

    uint64_t geometry_id = 0;
    scalar local0 = 0.;
    scalar local1 = 0.;
    scalar var_local0 = 0.;
    scalar var_local1 = 0.;

    // geometry_id,hit_id,channel0,channel1,timestamp,value
    DFE_NAMEDTUPLE(csv_measurement, geometry_id, local0, local1, var_local0, var_local1);

  };

  using measurement_writer = dfe::NamedTupleCsvWriter<csv_measurement>;
  
  struct csv_spacepoint {

    uint64_t geometry_id = 0;
    scalar x, y, z;
    scalar var_x, var_y, var_z;

    // geometry_id,hit_id,channel0,channel1,timestamp,value
    DFE_NAMEDTUPLE(csv_spacepoint, geometry_id, x, y, z, var_x, var_y, var_z);

  };

  using spacepoint_writer = dfe::NamedTupleCsvWriter<csv_spacepoint>;

  struct csv_surface {
    
    uint64_t geometry_id = 0;
    scalar cx, cy, cz;
    scalar rot_xu,rot_xv,rot_xw;
    scalar rot_yu,rot_yv,rot_yw;
    scalar rot_zu,rot_zv,rot_zw;

    // geometry_id,hit_id,channel0,channel1,timestamp,value
    DFE_NAMEDTUPLE(csv_surface, geometry_id, cx,cy,cz,rot_xu,rot_xv,rot_xw,rot_yu,rot_yv,rot_yw,rot_zu,rot_zv,rot_zw);

  };

  using surface_reader = dfe::NamedTupleCsvReader<csv_surface>;

  /// Read the geometry information per module and fill into a map
  ///
  /// @param sreader The surface reader type
  std::map<geometry_id, transform3> read_surfaces(surface_reader& sreader){

    std::map<geometry_id, transform3> transform_map;
    csv_surface iosurface;
    while (sreader.read(iosurface)){

      geometry_id module = iosurface.geometry_id;
      
      vector3 t{iosurface.cx, iosurface.cy, iosurface.cz};
      vector3 x{iosurface.rot_xu, iosurface.rot_yu, iosurface.rot_zu};
      vector3 z{iosurface.rot_xw, iosurface.rot_yw, iosurface.rot_zw};

      transform_map.insert({module,transform3{t,z,x}});

    }
    return transform_map;

  }

  /// Read the collection of cells per module and fill into a collection
  /// 
  /// @param creader The cellreader type
  /// @param tfmap the (optional) transform map
  /// @param max_cells the (optional) maximum number of cells to be read in
  std::vector<cell_collection> read_cells(cell_reader& creader, 
            const std::map<geometry_id, transform3>& tfmap ={}, 
                unsigned int max_cells = std::numeric_limits<unsigned int>::max()){

    uint64_t reference_id = 0;
    std::vector<cell_collection> cell_container;

    bool first_line_read = false;
    unsigned int read_cells = 0;
    csv_cell iocell;
    cell_collection cells;
    while (creader.read(iocell)){

      if (first_line_read and iocell.geometry_id != reference_id){
        // Complete the information
        if (not tfmap.empty()){
          auto tfentry = tfmap.find(iocell.geometry_id);
          if (tfentry != tfmap.end()){
             cells.placement = tfentry->second;
          }
        }
        // Sort in column major order
        std::sort(cells.items.begin(), cells.items.end(), [](const auto& a, const auto& b){ return a.channel1 < b.channel1; } );
        cell_container.push_back(cells);
        // Clear for next round
        cells = cell_collection();
      }
      first_line_read = true;
      reference_id = static_cast<uint64_t>(iocell.geometry_id);

      cells.module = reference_id;
      cells.range0[0] = std::min(cells.range0[0],iocell.channel0);
      cells.range0[1] = std::max(cells.range0[1],iocell.channel0);
      cells.range1[0] = std::min(cells.range1[0],iocell.channel1);
      cells.range1[1] = std::max(cells.range1[1],iocell.channel1);

      cells.items.push_back(cell{iocell.channel0, iocell.channel1, iocell.value, iocell.timestamp});
      if (++read_cells >= max_cells){
        break;
      }
    }    

    // Clean up after loop
    // Sort in column major order
    std::sort(cells.items.begin(), cells.items.end(), [](const auto& a, const auto& b){ return a.channel1 < b.channel1; } );
    cell_container.push_back(cells);

    return cell_container;
  }

  /// Read the collection of cells per module and fill into a collection
  /// of truth clusters.
  /// 
  /// @param creader The cellreader type
  /// @param tfmap the (optional) transform map
  /// @param max_clusters the (optional) maximum number of cells to be read in
  std::vector<cluster_collection> read_truth_clusters(cell_reader& creader, 
            const std::map<geometry_id, transform3>& tfmap ={}, 
                unsigned int max_cells = std::numeric_limits<unsigned int>::max()){

    // Reference for switching the container
    uint64_t reference_id = 0;
    std::vector<cluster_collection> cluster_container;
    // Reference for switching the cluster
    uint64_t truth_id = std::numeric_limits<uint64_t>::max();

    bool first_line_read = false;
    unsigned int read_cells = 0;
    csv_cell iocell;
    cluster_collection truth_clusters;
    std::vector<cell> truth_cells;

    while (creader.read(iocell)){

      if (first_line_read and iocell.geometry_id != reference_id){
        // Complete the information
        if (not tfmap.empty()){
          auto tfentry = tfmap.find(iocell.geometry_id);
          if (tfentry != tfmap.end()){
             truth_clusters.placement = tfentry->second;
          }
        }

        // Sort in column major order
        cluster_container.push_back(truth_clusters);
        // Clear for next round
        truth_clusters = cluster_collection();
      }

      if (first_line_read and truth_id != iocell.hit_id){
          truth_clusters.items.push_back({truth_cells});
          truth_cells.clear();
      } 
      truth_cells.push_back(cell{iocell.channel0, iocell.channel1, iocell.value, iocell.timestamp});

      first_line_read = true;
      truth_id = iocell.hit_id;
      reference_id = static_cast<uint64_t>(iocell.geometry_id);

      if (++read_cells >= max_cells){
        break;
      }
    }    

    return cluster_container;
  }


}
