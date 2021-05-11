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
#include "algorithms/seeding/bin_finder.hpp"
#include "algorithms/seeding/seedfinder_config.hpp"
#include "algorithms/seeding/internal_spacepoint.hpp"

namespace traccc{

/// Iterates over the elements of all bins given
/// by the indices parameter in the given SpacePointGrid.
/// Fullfills the forward iterator.
class neighborhood_iterator {
 public:
    //using sp_it_t = typename std::vector<std::unique_ptr<
    //const internal_spacepoint<spacepoint>>>::const_iterator;
    using sp_it_t = typename vecmem::vector< internal_spacepoint<spacepoint>>::const_iterator;
    
    neighborhood_iterator() = delete;
    
    neighborhood_iterator(std::vector<size_t> indices,
			  const spacepoint_grid* spgrid) {
	m_grid = spgrid;
	m_indices = indices;
	m_curInd = 0;
	if (m_indices.size() > m_curInd) {
	    m_curIt = std::begin(spgrid->at(m_indices[m_curInd]));
	    m_binEnd = std::end(spgrid->at(m_indices[m_curInd]));
	}
    }
    
    neighborhood_iterator(std::vector<size_t> indices,
			  const spacepoint_grid* spgrid,
			  size_t curInd, sp_it_t curIt) {
	m_grid = spgrid;
	m_indices = indices;
	m_curInd = curInd;
	m_curIt = curIt;
	if (m_indices.size() > m_curInd) {
	    m_binEnd = std::end(spgrid->at(m_indices[m_curInd]));
	}
    }
    static neighborhood_iterator begin(
				       std::vector<size_t> indices,
				       const spacepoint_grid* spgrid) {
	auto nIt = neighborhood_iterator(indices, spgrid);
	// advance until first non-empty bin or last bin
	if (nIt.m_curIt == nIt.m_binEnd) {
	    ++nIt;
	}
	return nIt;
    }
    
    neighborhood_iterator(
			  const neighborhood_iterator& other) {
	m_grid = other.m_grid;
	m_indices = other.m_indices;
	m_curInd = other.m_curInd;
	m_curIt = other.m_curIt;
	m_binEnd = other.m_binEnd;
    }
    
    void operator++() {
	// if iterator of current Bin not yet at end, increase
	if (m_curIt != m_binEnd) {
	    m_curIt++;
	    // return only if end of current bin still not reached
	    if (m_curIt != m_binEnd) {
		return;
	    }
	}
	// increase bin index m_curInd until you find non-empty bin
	// or until m_curInd >= m_indices.size()-1
	while (m_curIt == m_binEnd && m_indices.size() - 1 > m_curInd) {
	    m_curInd++;
	    m_curIt = std::begin(m_grid->at(m_indices[m_curInd]));
	    m_binEnd = std::end(m_grid->at(m_indices[m_curInd]));
	}
    }

    const internal_spacepoint<spacepoint> operator*() {
	return *m_curIt;
    }
    
    bool operator!=(const neighborhood_iterator& other) {
	return m_curIt != other.m_curIt || m_curInd != other.m_curInd;
    }

    // iterators within current bin
    sp_it_t m_curIt;
    sp_it_t m_binEnd;
    // number of bins
    std::vector<size_t> m_indices;
    // current bin
    size_t m_curInd;
    const spacepoint_grid* m_grid;
};

    
///@class Neighborhood Used to access iterators to access a group of bins
/// returned by a BinFinder.
/// Fulfills the range_expression interface
class neighborhood {
public:
    neighborhood() = delete;
    neighborhood(std::vector<size_t> indices,
		 const spacepoint_grid* spgrid) {
	m_indices = indices;
	m_spgrid = spgrid;
    }
    
    neighborhood_iterator begin() {
	return neighborhood_iterator::begin(m_indices,
					    m_spgrid);
    }
    neighborhood_iterator end() {
	return neighborhood_iterator(
				     m_indices, m_spgrid, m_indices.size() - 1,
				     std::end(m_spgrid->at(m_indices.back())) );
    }
    
 private:
  std::vector<size_t> m_indices;
  const spacepoint_grid* m_spgrid;
};
    
///@class BinnedSPGroupIterator Allows to iterate over all groups of bins
/// a provided BinFinder can generate for each bin of a provided SPGrid
class binned_spgroup_iterator {
public:
    binned_spgroup_iterator& operator++() {
	if (zIndex < phiZbins[1]) {
	    zIndex++;
	    
	} else {
	    zIndex = 1;
	    phiIndex++;
	}
	// set current & neighbor bins only if bin indices valid
	if (phiIndex <= phiZbins[0] && zIndex <= phiZbins[1]) {
	    currentBin =
		std::vector<size_t>{grid->globalBinFromLocalBins({phiIndex, zIndex})};
	    bottomBinIndices = m_bottomBinFinder->findBins(phiIndex, zIndex, grid);
	    topBinIndices = m_topBinFinder->findBins(phiIndex, zIndex, grid);
	    outputIndex++;
	    return *this;
	}
	phiIndex = phiZbins[0];
	zIndex = phiZbins[1] + 1;
	return *this;
    }

    bool operator==(const binned_spgroup_iterator& otherState) {
	return (zIndex == otherState.zIndex && phiIndex == otherState.phiIndex);
    }
    
    bool operator!=(const binned_spgroup_iterator& otherState) {
	return !(this->operator==(otherState));
    }
    
    neighborhood middle() {
	return neighborhood(currentBin, grid);
    }
    
    neighborhood bottom() {
	return neighborhood(bottomBinIndices, grid);
    }
    
    neighborhood top() {
	return neighborhood(topBinIndices, grid);
    }
   
    binned_spgroup_iterator(const spacepoint_grid* spgrid,
			    bin_finder* botBinFinder,
			    bin_finder* tBinFinder)
	: currentBin({spgrid->globalBinFromLocalBins({1, 1})}) {
	grid = spgrid;
	m_bottomBinFinder = botBinFinder;
	m_topBinFinder = tBinFinder;
	phiZbins = grid->numLocalBins();
	phiIndex = 1;
	zIndex = 1;
	outputIndex = 0;
	bottomBinIndices = m_bottomBinFinder->findBins(phiIndex, zIndex, grid);
	topBinIndices = m_topBinFinder->findBins(phiIndex, zIndex, grid);
    }
    
    binned_spgroup_iterator(const spacepoint_grid* spgrid,
			    bin_finder* botBinFinder,
			    bin_finder* tBinFinder,
			    size_t phiInd, size_t zInd)
	: currentBin({spgrid->globalBinFromLocalBins({phiInd, zInd})}) {
	m_bottomBinFinder = botBinFinder;
	m_topBinFinder = tBinFinder;
	grid = spgrid;
	phiIndex = phiInd;
	zIndex = zInd;
	phiZbins = grid->numLocalBins();
	outputIndex = (phiInd - 1) * phiZbins[1] + zInd - 1;
	if (phiIndex <= phiZbins[0] && zIndex <= phiZbins[1]) {
	    bottomBinIndices = m_bottomBinFinder->findBins(phiIndex, zIndex, grid);
	    topBinIndices = m_topBinFinder->findBins(phiIndex, zIndex, grid);
	}
    }
    
 private:
    // middle spacepoint bin
    std::vector<size_t> currentBin;
    std::vector<size_t> bottomBinIndices;
    std::vector<size_t> topBinIndices;
    const spacepoint_grid* grid;
    size_t phiIndex = 1;
    size_t zIndex = 1;
    size_t outputIndex = 0;
    std::array<long unsigned int, 2ul> phiZbins;
    bin_finder* m_bottomBinFinder;
    bin_finder* m_topBinFinder;
};


class binned_spgroup{
public:
        
    binned_spgroup(const host_spacepoint_container& sp_container,
		   spacepoint_grid& sp_grid,
		   const seedfinder_config& config):
	m_binned_sp(std::move(sp_grid)){    
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
	std::vector<std::vector< internal_spacepoint< spacepoint > > > rBins(numRBins);
	
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
		
		auto isp = internal_spacepoint<spacepoint> (sp, spPosition, config.beamPos, variance);
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
	for (auto& rbin : rBins) {
	    for (auto& isp : rbin) {
		vector2 spLocation({isp.phi(), isp.z()});
		vecmem::vector< internal_spacepoint< spacepoint > >& bin = m_binned_sp.at_position(spLocation);
		bin.push_back(std::move(isp));	    
	    }
	}
	
    }

    size_t size() { return m_binned_sp.size(); }

    binned_spgroup_iterator begin() {
	return binned_spgroup_iterator(&m_binned_sp, m_bottomBinFinder.get(), m_topBinFinder.get());
    }

    binned_spgroup_iterator end() {
	auto phiZbins = m_binned_sp.numLocalBins();
	return binned_spgroup_iterator(
            &m_binned_sp, m_bottomBinFinder.get(), m_topBinFinder.get(),
            phiZbins[0], phiZbins[1] + 1);
    }
    
private:
    spacepoint_grid m_binned_sp;
    // BinFinder must return std::vector<Acts::Seeding::Bin> with content of
    // each bin sorted in r (ascending)
    std::shared_ptr<bin_finder > m_topBinFinder;
    std::shared_ptr<bin_finder > m_bottomBinFinder;
    
};

} // namespace traccc
