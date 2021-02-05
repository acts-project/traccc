/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/cell.hpp"

#include <dfe/dfe_namedtuple.hpp>
#include <dfe/dfe_io_dsv.hpp>

#include <fstream>

namespace traccc {

  struct csv_cell {
    
    geometry_id module_id  = 0;
    channel_id channel0 = 0;
    channel_id channel1 = 0;
    scalar activation = 0.;
    scalar time = 0.;

    DFE_NAMEDTUPLE(csv_cell, module_id, channel0, channel1, activation, time);

  };

  using cell_reader = dfe::NamedTupleCsvReader<csv_cell>;

  /// Read the collection for a range per module module,
  /// it stops reading when the module identification changes.
  /// 
  /// @param creader The cellreader type
  cell_collection read_cells_per_module(cell_reader& creader){

    int reference_id = -1;

    csv_cell iocell;
    cell_collection cells;
    while (creader.read(iocell)){

      if (reference_id >= 0 and iocell.module_id != reference_id){
        break;
      } 
      reference_id = static_cast<int>(iocell.module_id);

      cells.module_id = iocell.module_id;
      cells.range0[0] = std::min(cells.range0[0],iocell.channel0);
      cells.range0[1] = std::max(cells.range0[1],iocell.channel0);
      cells.range1[0] = std::min(cells.range1[0],iocell.channel1);
      cells.range1[1] = std::max(cells.range1[1],iocell.channel1);

      cells.items.push_back(cell{iocell.channel0, iocell.channel1, iocell.activation, iocell.time});
    }    

    // Sort in column major order
    std::sort(cells.items.begin(), cells.items.end(), [](const auto& a, const auto& b){ return a.channel1 < b.channel1; } );

    return cells;
  }


}
