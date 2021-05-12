/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/neighborhood_index.hpp>
#include <edm/internal_spacepoint.hpp>
#include <edm/spacepoint.hpp>
#include <algorithms/seeding/seeding_config.hpp>
#include <algorithm>

namespace traccc{

    // group the spacepoints basaed on its position
    struct spacepoint_grouping{

	// constructor declaration
	spacepoint_grouping(std::shared_ptr<spacepoint_grid> spgrid, const seedfinder_config& config)
	    : m_spgrid(spgrid),
	      m_config(config) {};

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
	    vecmem::vector<size_t> unique_idx;
	    auto& headers = internal_sp_container.headers;
	    auto& items = internal_sp_container.items;
	    
	    for (auto& rbin : rBins) {
		for (auto& isp : rbin) {
		    vector2 spLocation({isp.phi(), isp.z()});
		    auto global_bin = m_spgrid->globalBinFromPosition(spLocation);

		    if (std::find(headers.begin(), headers.end(), global_bin) == headers.end()){
			headers.push_back(global_bin);
			items.push_back(vecmem::vector<internal_spacepoint<spacepoint>>());
		    }
		    auto container_location = std::find(headers.begin(), headers.end(), global_bin) - headers.begin();
		    items.at(container_location).push_back(std::move(isp));
		}
	    }
	}
		       	
    private:
	seedfinder_config m_config;
	std::shared_ptr< spacepoint_grid > m_spgrid;
    };
	
}// namespace traccc
