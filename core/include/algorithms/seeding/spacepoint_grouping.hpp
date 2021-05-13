/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/internal_spacepoint.hpp>
#include <edm/spacepoint.hpp>
#include <algorithms/seeding/seeding_config.hpp>
#include <algorithms/seeding/bin_finder.hpp>
#include <algorithm>

namespace traccc{

    // group the spacepoints basaed on its position
    struct spacepoint_grouping{

	// constructor declaration
	spacepoint_grouping(const seedfinder_config& config, const spacepoint_grid_config& grid_config);

	host_internal_spacepoint_container operator()(const host_spacepoint_container& sp_container){
	    host_internal_spacepoint_container internal_sp_container;

	    this->operator()(sp_container, internal_sp_container);

	    return internal_sp_container;
	}

	void operator()(const host_spacepoint_container& sp_container,
			host_internal_spacepoint_container& internal_sp_container){
	    // get region of interest (or full detector if configured accordingly)
	    float phiMin = m_config.phiMin;
	    float phiMax = m_config.phiMax;
	    float zMin = m_config.zMin;
	    float zMax = m_config.zMax;
	    
	    // sort by radius
	    // add magnitude of beamPos to rMax to avoid excluding measurements
	    // create number of bins equal to number of millimeters rMax
	    // (worst case minR: m_configured minR + 1mm)
	    
	    size_t numRBins = (m_config.rMax + getter::norm(m_config.beamPos));
	    std::vector< std::vector< internal_spacepoint<spacepoint> > > rBins(numRBins);
	    
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
		    // vector2 variance = covTool(sp, m_config.zAlign, m_config.rAlign, m_config.sigmaError);
		    vector2 variance({varR, varZ});
		    vector3 spPosition({spX, spY, spZ});
		    
		    auto isp = internal_spacepoint<spacepoint> (sp, spPosition, m_config.beamPos, variance);
		    // calculate r-Bin index and protect against overflow (underflow not
		    // possible)
		    size_t rIndex = isp.radius();
		    // if index out of bounds, the SP is outside the region of interest
		    if (rIndex >= numRBins) {
			continue;
		    }
		    rBins[rIndex].push_back(std::move(isp));	    
		}
	    }
	    
	    // fill rbins into grid such that each grid bin is sorted in r
	    // space points with delta r < rbin size can be out of order
	    auto& headers = internal_sp_container.headers;
	    auto& items = internal_sp_container.items;


	    for (auto& rbin : rBins) {
		for (auto& isp : rbin) {
		    vector2 spLocation({isp.phi(), isp.z()});
		    auto local_bin = m_spgrid->localBinsFromPosition(spLocation);
		    auto global_bin = m_spgrid->globalBinFromLocalBins(local_bin);

		    // check if the global bin has not been recorded		    
		    auto it = std::find_if(headers.begin(), headers.end(),
					   [&global_bin](const bin_info& binfo)
					   {return binfo.global_index==global_bin;});
		    
		    if (it == std::end(headers) ){
			auto bottom_indices = m_bottom_bin_finder->find_bins(local_bin[0], local_bin[1], m_spgrid.get());
			auto top_indices = m_top_bin_finder->find_bins(local_bin[0], local_bin[1], m_spgrid.get());
			
			bin_info binfo;			
			binfo.global_index = global_bin;			
			binfo.num_bottom_bin_indices = bottom_indices.size();
			binfo.num_top_bin_indices = top_indices.size();

			std::copy(bottom_indices.begin(), bottom_indices.end(), &binfo.bottom_bin_indices[0]);
			std::copy(top_indices.begin(), top_indices.end(), &binfo.top_bin_indices[0]);			
			
			headers.push_back(binfo);
			items.push_back(vecmem::vector<internal_spacepoint<spacepoint>>());
		    }

		    auto container_location
			= std::find_if(headers.begin(), headers.end(),
				       [&global_bin](const bin_info& binfo)
				       {return binfo.global_index==global_bin;})
			- headers.begin();

		    items.at(container_location).push_back(std::move(isp));    

		}
	    }
	}
		       	
    private:
	seedfinder_config m_config;
	spacepoint_grid_config m_grid_config;
	std::shared_ptr< spacepoint_grid > m_spgrid;
	std::unique_ptr< bin_finder > m_bottom_bin_finder;
	std::unique_ptr< bin_finder > m_top_bin_finder;	
    };

    spacepoint_grouping::spacepoint_grouping(const seedfinder_config& config, const spacepoint_grid_config& grid_config)
	: m_config(config),
	  m_grid_config(grid_config){
	
	// calculate circle intersections of helix and max detector radius
	float minHelixRadius = grid_config.minPt / (300. * grid_config.bFieldInZ);  // in mm
	float maxR2 = grid_config.rMax * grid_config.rMax;
	float xOuter = maxR2 / (2 * minHelixRadius);
	float yOuter = std::sqrt(maxR2 - xOuter * xOuter);
	float outerAngle = std::atan(xOuter / yOuter);
	// intersection of helix and max detector radius minus maximum R distance from
	// middle SP to top SP
	float innerAngle = 0;
	if (grid_config.rMax > grid_config.deltaRMax) {
	    float innerCircleR2 =
		(grid_config.rMax - grid_config.deltaRMax) * (grid_config.rMax - grid_config.deltaRMax);
	    float xInner = innerCircleR2 / (2 * minHelixRadius);
	    float yInner = std::sqrt(innerCircleR2 - xInner * xInner);
	    innerAngle = std::atan(xInner / yInner);
	}
	
	// FIXME: phibin size must include max impact parameters
	// divide 2pi by angle delta to get number of phi-bins
	// size is always 2pi even for regions of interest
	int phiBins = std::floor(2 * M_PI / (outerAngle - innerAngle));
	axis<AxisBoundaryType::Closed> phiAxis(-M_PI, M_PI, phiBins);
	
	// TODO: can probably be optimized using smaller z bins
	// and returning (multiple) neighbors only in one z-direction for forward
	// seeds
	// FIXME: zBinSize must include scattering
	
	float zBinSize = grid_config.cotThetaMax * grid_config.deltaRMax;
	int zBins = std::floor((grid_config.zMax - grid_config.zMin) / zBinSize);
	axis<AxisBoundaryType::Bound> zAxis(grid_config.zMin, grid_config.zMax, zBins);

	m_spgrid = std::make_shared<spacepoint_grid>(spacepoint_grid(std::make_tuple(phiAxis, zAxis)));	

    }    
}// namespace traccc
