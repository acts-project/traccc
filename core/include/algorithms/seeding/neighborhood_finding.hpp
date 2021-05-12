/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/neighborhood_index.hpp>
#include <algorithms/seeding/seeding_config.hpp>

namespace traccc{
    
    /// @class bin_finder
    /// The bin_finder is used by the binned_spgroup. It can be
    /// used to find both bins that could be bottom bins as well as bins that could
    /// be top bins, which are assumed to be the same bins. Does not take
    /// interaction region into account to limit z-bins.
    class bin_finder {
    public:
	/// destructor
	~bin_finder() = default;
	
	/// Return all bins that could contain space points that can be used with the
	/// space points in the bin with the provided indices to create seeds.
	/// @param phiBin phi index of bin with middle space points
	/// @param zBin z index of bin with middle space points
	/// @param binnedSP phi-z grid containing all bins
	vecmem::vector<size_t> find_bins(size_t phi_bin, size_t z_bin,
				      const spacepoint_grid* binned_sp){
	    return binned_sp->neighborhood_indices({phi_bin, z_bin}).collect();
	}
    };
        
    // Find bottom and top bins
    struct neighborhood_finding{

	neighborhood_finding(const spacepoint_grid_config& config);
	    	
	std::pair< host_neighborhood_index_container,
		   host_neighborhood_index_container > operator()(){
	    
	    host_neighborhood_index_container bot_indices;
	    host_neighborhood_index_container top_indices;

	    this->operator()(bot_indices, top_indices);
	    return std::make_pair(bot_indices, top_indices);
	}

	void operator()(host_neighborhood_index_container& bot_indices,
			host_neighborhood_index_container& top_indices) const{
	    
	    auto phiZbins = m_spgrid->numLocalBins();

	    int i =0;
	    
	    for (size_t phi_index = 1; phi_index <= phiZbins[0]; ++phi_index){
		for (size_t z_index = 1; z_index <= phiZbins[1]; ++z_index){

		    auto bot_bin_indices = m_bot_bin_finder->find_bins(phi_index, z_index, m_spgrid.get());
		    auto top_bin_indices = m_top_bin_finder->find_bins(phi_index, z_index, m_spgrid.get());
		    bot_indices.headers.push_back(bot_bin_indices.size());
		    bot_indices.items.push_back(bot_bin_indices);

		    top_indices.headers.push_back(top_bin_indices.size());
		    top_indices.items.push_back(top_bin_indices);
		}
	    }
	}

	std::shared_ptr<spacepoint_grid> get_grid(){return m_spgrid;}
	
    private:
	std::shared_ptr< spacepoint_grid > m_spgrid;
	std::unique_ptr< bin_finder > m_bot_bin_finder;
	std::unique_ptr< bin_finder > m_top_bin_finder;
    };

    // constructor implementation
    neighborhood_finding::neighborhood_finding(const spacepoint_grid_config& config){
	
	// calculate circle intersections of helix and max detector radius
	float minHelixRadius = config.minPt / (300. * config.bFieldInZ);  // in mm
	float maxR2 = config.rMax * config.rMax;
	float xOuter = maxR2 / (2 * minHelixRadius);
	float yOuter = std::sqrt(maxR2 - xOuter * xOuter);
	float outerAngle = std::atan(xOuter / yOuter);
	// intersection of helix and max detector radius minus maximum R distance from
	// middle SP to top SP
	float innerAngle = 0;
	if (config.rMax > config.deltaRMax) {
	    float innerCircleR2 =
		(config.rMax - config.deltaRMax) * (config.rMax - config.deltaRMax);
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
	
	float zBinSize = config.cotThetaMax * config.deltaRMax;
	int zBins = std::floor((config.zMax - config.zMin) / zBinSize);
	axis<AxisBoundaryType::Bound> zAxis(config.zMin, config.zMax, zBins);

	m_spgrid = std::make_shared<spacepoint_grid>(spacepoint_grid(std::make_tuple(phiAxis, zAxis)));
    }
    
}// namespace traccc
