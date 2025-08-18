/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

//System include(s)
#include <memory>

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

//Detray include(s).
#include <detray/geometry/barcode.hpp>

namespace traccc::device {
//where to put these definitions?
struct gbts_layerInfo {
	//int2 vector type
	int etaBin0{};
	int numBins{};
	//float2 vector type
	float minEta{};
	float delatEta{};
	bool isEndcap{};
};

enum class gbts_consts : unsigned short {

	//for GPU seed extraction
	shared_state_buffer_size = 578,
	//matrix access for kalman filter state
	M3_0_0 = 0, 
	M3_0_1 = 1, 
	M3_0_2 = 2,
	M3_1_1 = 3,
	M3_1_2 = 4,
	M3_2_2 = 5,

	M2_0_0 = 0,
	M2_0_1 = 1,
	M2_1_1 = 1,
}; 
}

namespace traccc {

struct gbts_seedfinder_config {
	gbts_seedfinder_config(const std::vector<std::pair<int, int>>& binTables, const std::vector<device::GBTS::gbts_layerInfo>& layerInfo,
              const std::vector<std::pair<uint64_t, int>>& detrayBarcodeBinning, float minPt);

	//layer linking and geometry	
	std::vector<std::pair<int, int>> binTables{};
	std::vector<device::GBTS::gbts_layerInfo> layerInfo{};
	unsigned long int nLayers = 0;	

	std::shared_ptr<int[]> volumeToLayerMap{};
	unsigned long int maxVolIndex = 0;	

	std::vector<std::array<unsigned int, 2>> surfaceToLayerMap;
	unsigned long int surfaceMapSize = 0;	

	//tuned for 900 MeV pT cut and scaled by input minPt	
	//edge making cuts
	float min_deltaPhi = 0.015f;
	float dphi_coeff = 2.2e-4f;
	float min_deltaPhi_low_dr = 0.002f;
	float dphi_coeff_low_dr = 4.33e-4f;	
	float minDeltaRadius = 2.0f;
	float min_z0 = 160.0f;
	float max_z0 = 160.0f;
	float maxOuterRadius = 550.0f; //change to 350
	float cut_zMinU = min_z0 - maxOuterRadius*36;
	float cut_zMaxU = max_z0 + maxOuterRadius*36; //how to get ROI dzdr
	float maxKappa = 0.337f;
	float low_Kappa_d0 = 0.02f;
	float high_Kappa_d0 = 0.1f;

	//edge matching cuts
	float cut_dphi_max = 0.012f;
	float cut_dcurv_max = 0.001f;
	float cut_tau_ratio_max = 0.01f;

	//graph making maxiums
	unsigned char max_num_neighbours = 10;
	unsigned short node_buffer_length = 250;
	unsigned short max_phi_bin_size = 120;
	
	//seed extraction maxiums
	unsigned char max_cca_iterations = 20;
};

//binTables contains pairs of linked layer-eta bins
//the layerInfo should be calculated from the barcodeBinning
//BarcodeBinning pair is detray barcode and bin index (corrisponding to the layers in layerInfo) ordered by volume 
//minPt in MeV
gbts_seedfinder_config::gbts_seedfinder_config(const std::vector<std::pair<int, int>>& input_binTables, const std::vector<device::GBTS::gbts_layerInfo>& input_layerInfo,
	  const std::vector<std::pair<uint64_t, int>>& detrayBarcodeBinning, float minPt = 900.0f) {

	//format layer-eta binning infomation
	binTables = input_binTables;
	layerInfo = input_layerInfo;

	int current_volume = 0;

	int current_layer = 0;
	bool layerChange = false;

	std::pair<uint64_t, int> surfaceLayerPair;
	std::vector<std::array<unsigned int, 2>> surfacesInVolume;

	std::vector<std::pair<int, int>> volumeToLayerMap_unordered;
	unsigned int largest_volume_index = 0;
	for(int index = 0; index<detrayBarcodeBinning.size(); index++) {

		surfaceLayerPair = detrayBarcodeBinning[index];
		detray::geometry::barcode barcode(surfaceLayerPair.first);

		if(current_layer == -1) current_layer = surfaceLayerPair.second;
		if(current_volume == -1) current_volume = barcode.volume();

		//save surfaces incase volume is not encommpassed by a layer
		surfacesInVolume.push_back(std::array<unsigned int, 2>(static_cast<unsigned int>(barcode.id()), static_cast<unsigned int>(surfaceLayerPair.second)));

		//is volume encompassed by a layer
		if(current_layer != surfaceLayerPair.second) layerChange = true;

		//create volume veiw on the surface map
		if(barcode.volume() != current_volume) {

			int bin = -1*surfaceLayerPair.second;
			if(layerChange) {
				bin = static_cast<int>(surfaceToLayerMap.size() + 1); //start of this volumes surfaces in the map
				for(std::array<unsigned int, 2> pair : surfacesInVolume) surfaceToLayerMap.push_back(pair);
			}
			volumeToLayerMap_unordered.push_back(std::make_pair(bin, static_cast<int>(barcode.volume()))); // -ve layerIdx if one-to-one +ve index + 1 otherwise
			if(barcode.volume() > largest_volume_index) largest_volume_index = barcode.volume();
			current_layer  = -1;
			current_volume = -1;
			layerChange = false;
		}
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
	maxVolIndex = largest_volume_index;
	nLayers     = layerInfo.size();
	surfaceMapSize   = surfaceToLayerMap.size();
}
} //namespace traccc
