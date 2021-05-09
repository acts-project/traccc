/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/spacepoint.hpp"
#include "definitions/algebra.hpp"
#include "algorithms/seeding/spacepoint_grid.hpp"
#include "algorithms/seeding/seedfinder_config.hpp"
#include "algorithms/seeding/internal_spacepoint.hpp"

namespace traccc{
    
class binned_spgroup_iterator{
private:
    
};

class binned_spgroup{
public:
        
binned_spgroup(const host_spacepoint_container& sp_container,
	       spacepoint_grid& sp_grid,
	       const seedfinder_config& config){
    // get region of interest (or full detector if configured accordingly)
    float phiMin = config.phiMin;
    float phiMax = config.phiMax;
    float zMin = config.zMin;
    float zMax = config.zMax;
    
    // sort by radius
    // add magnitude of beamPos to rMax to avoid excluding measurements
    // create number of bins equal to number of millimeters rMax
    // (worst case minR: configured minR + 1mm)
    
    size_t numRBins = (config.rMax + getter::norm(config.beamPos));
    std::vector<std::vector< std::unique_ptr< const internal_spacepoint< spacepoint > > > > rBins(numRBins);
    
    for (auto& sp_vec: sp_container.items){
	for(auto& sp: sp_vec){
		    
	    float spX = sp.global[0];
	    float spY = sp.global[1];
	    float spZ = sp.global[2];
	    float varR = sp.variance[0]; // Need a check
	    float varZ = sp.variance[1];
	    
	    if (spZ > zMax || spZ < zMin) {
		continue;
	    }
	    float spPhi = std::atan2(spY, spX);
	    if (spPhi > phiMax || spPhi < phiMin) {
		continue;
	    }
	    	    
	    // Note: skip covTool of ACTS main repository
	    // vector2 variance = covTool(sp, config.zAlign, config.rAlign, config.sigmaError);
	    vector2 variance({varR, varZ});
	    vector3 spPosition({spX, spY, spZ});
	    auto isp =
		std::make_unique<const internal_spacepoint<spacepoint>>(
									sp, spPosition, config.beamPos, variance);
	    // calculate r-Bin index and protect against overflow (underflow not
	    // possible)
	    size_t rIndex = isp->radius();
	    // if index out of bounds, the SP is outside the region of interest
	    if (rIndex >= numRBins) {
		continue;
	    }
	    rBins[rIndex].push_back(std::move(isp));	    
	}
    }

    // fill rbins into grid such that each grid bin is sorted in r
    // space points with delta r < rbin size can be out of order
    for (auto& rbin : rBins) {
	for (auto& isp : rbin) {
	    vector2 spLocation({isp->phi(), isp->z()});
	    
	    std::vector<
		std::unique_ptr<const internal_spacepoint< spacepoint >>>&
		bin = sp_grid.at_position(spLocation);
	    bin.push_back(std::move(isp));
	    
	}
    }    
    //m_binned_sp = 
}

private:
    vecmem::jagged_vector<spacepoint> m_binned_sp;
};

} // namespace traccc
