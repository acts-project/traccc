/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/gbts_seeding/gbts_seeding_config.hpp"

namespace traccc {

//binTables contains pairs of linked layer-eta bins
//the layerInfo should really be calculated from the barcodeBinning
//BarcodeBinning pair is detray barcode and bin index (corrisponding to the layers in layerInfo) 
//minPt in MeV
gbts_seedfinder_config::gbts_seedfinder_config(const std::vector<std::pair<int, std::vector<int>>>& input_binTables, const device::gbts_layerInfo& input_layerInfo,
	  const std::vector<std::vector<std::pair<uint64_t, int>>>& detrayBarcodeBinning, float minPt = 900.0f) {

	//copy layer-eta binning infomation
	layerInfo = input_layerInfo;
	// unroll binTables
	for(std::pair<int, std::vector<int>> binPairs : input_binTables) {
		for(int bin2 : binPairs) binTables.push_back(std::make_pair<binPairs.first, bin2>);
	}
	
	bool layerChange = false;

	std::vector<std::pair<int, int>> volumeToLayerMap_unordered;
	unsigned int largest_volume_index = 0;
	for(std::vector<std::pair<uint64_t, int>> barcodeBin : detrayBarcodeBinning) {
		int current_layer = barcodesBin[0].second;
		std::vector<std::array<unsigned int, 2>> surfacesInVolume;
		surfacesInVolume.reserve(barcodesBin.size());
		for(std::pair<uint64_t, int> surfaceLayerPair : barcodeBin) {
			detray::geometry::barcode barcode(surfaceLayerPair.first);

			//save surfaces incase volume is not encommpassed by a layer
			surfacesInVolume.push_back(std::array<unsigned int, 2>{static_cast<unsigned int>(barcode.id()), static_cast<unsigned int>(surfaceLayerPair.second)});

			//is volume encompassed by a layer
			if(current_layer != surfaceLayerPair.second) layerChange = true;
		}
		
		//create volume veiw on the surface map
		int bin = -1*surfaceLayerPair.second;
		if(layerChange) {
			bin = static_cast<int>(surfaceToLayerMap.size() + 1); //start of this volumes surfaces in the map
			for(std::array<unsigned int, 2> pair : surfacesInVolume) surfaceToLayerMap.push_back(pair);
		}
		volumeToLayerMap_unordered.push_back(std::make_pair(bin, current_volume)); // -ve layerIdx if one-to-one +ve index + 1 otherwise
		if(barcode.volume() > largest_volume_index) largest_volume_index = barcode.volume();
		current_layer  = -1;
		current_volume = -1;
		layerChange = false;
		
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
	nLayers        = layerInfo.size();
	surfaceMapSize = surfaceToLayerMap.size();
}

} //namespace traccc
