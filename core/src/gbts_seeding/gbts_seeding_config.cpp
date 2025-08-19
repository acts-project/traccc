/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/gbts_seeding/gbts_seeding_config.hpp"

#include <limits>
#include <algorithm>

namespace traccc {

//binTables contains pairs of linked layer-eta bins
//the layerInfo should really be calculated from the barcodeBinning
//BarcodeBinning pair is detray barcode and bin index (corrisponding to the layers in layerInfo) 
//minPt in MeV
bool gbts_seedfinder_config::setLinkingScheme(const std::vector<std::pair<int, std::vector<int>>>& input_binTables, const device::gbts_layerInfo& input_layerInfo,
	  const std::vector<std::pair<uint64_t, short>>& detrayBarcodeBinning, float minPt = 900.0f) {

	//copy layer-eta binning infomation
	layerInfo = input_layerInfo;
	// unroll binTables
	for(std::pair<int, std::vector<int>> binPairs : input_binTables) {
		for(int bin2 : binPairs) binTables.push_back(std::make_pair<binPairs.first, bin2>);
	}

	//bin by volume	
	std::sort(detrayBarcodeBinning, [](const std::pair<uint64_t, short> a, const std::pair<uint64_t, short> b) {return a.first > b.first;});

	bool layerChange     = false;
	short current_layer  = -1; 
	short current_volume = -1;	
	unsigned int largest_volume_index = 0;
	std::vector<std::pair<int, short>> volumeToLayerMap_unordered;

	for(std::pair<uint64_t, short> barcodeLayerPair : detrayBarcodeBinning) {
		std::vector<std::array<unsigned int, 2>> surfacesInVolume;
		
		detray::geometry::barcode barcode(barcodeLayerPair.first);
		if(current_volume == -1) current_volume = static_cast<short>(barcode.volume());		
		else if(current_volume != barcode.volume()) {	
			//reached the end of this volume so add it to the maps
			short bin = -1*current_layer;
			if(layerChange) {
				bin = static_cast<short>(surfaceToLayerMap.size() + 1); //start of this volume's surfaces in the map
				for(std::array<unsigned int, 2> pair : surfacesInVolume) surfaceToLayerMap.push_back(pair);
			}
			volumeToLayerMap_unordered.push_back(std::make_pair(bin, current_volume)); // -ve layerIdx if one-to-one +ve index + 1 otherwise
			if(current_volume > largest_volume_index) largest_volume_index = current_volume;
			current_layer  = -1;
			current_volume = -1;
			layerChange = false;
			surfacesInVolume.clear();
		}
		
		//is volume encompassed by a layer
		if(current_layer == -1) current_layer  = barcodeLayerPair.second;		
		else if(current_layer != barcodeLayerPair.second) {layerChange = true;}	

		//save surfaces incase volume is not encommpassed by a layer
		surfacesInVolume.push_back(std::array<unsigned int, 2>{static_cast<unsigned int>(barcode.id()), static_cast<unsigned int>(surfaceLayerPair.second)});
	}
	// make volume by layer map
	volumeToLayerMap = std::make_shared<int[]>(largest_volume_index);
	for(std::pair<int, unsigned int> vLpair : volumeToLayerMap_unordered) volumeToLayerMap[vLpair.second] = vLpair.first;

	//scale cuts
	float ptScale = 900.0f/minPt;
	min_deltaPhi*=ptScale;
	dphi_coeff*=ptScale;
	min_deltaPhi_low_dr*=ptScale;
	dphi_coeff_low_dr*=ptScale;
	maxKappa*=ptScale;

	//contianers sizes
	maxVolIndex    = largest_volume_index;
	nLayers        = layerInfo.isEndcap.size();
	surfaceMapSize = surfaceToLayerMap.size();
	if(surfaceMapSize > std::numeric_limits<short>::max) return false;
	return true;
}

} //namespace traccc
