/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/spacepoint.hpp"
#include "edm/binned_spacepoint.hpp"
#include "algorithms/seeding/spacepoint_grid.hpp"
#include "algorithms/seeding/seedfinder_config.hpp"

namespace traccc{

class binned_spgroup_iterator{
private:
    
};

class binned_spgroup{
public:
    
    
    binned_spgroup(const host_spacepoint_container& sp_container,
		   const spacepoint_grid& sp_grid,
		   const seedfinder_config& config,
		   host_binnedsp_container& binnedsp_container){
	// get region of interest (or full detector if configured accordingly)
	float phiMin = config.phiMin;
	float phiMax = config.phiMax;
	float zMin = config.zMin;
	float zMax = config.zMax;
	
	// sort by radius
	// add magnitude of beamPos to rMax to avoid excluding measurements
	// create number of bins equal to number of millimeters rMax
	// (worst case minR: configured minR + 1mm)
	
	//size_t numRBins = (config.rMax + config.beamPos.norm());
     
	//m_binned_sp = 
    }        
};

} // namespace traccc
