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
	std::vector<bool> isEndcap;
	std::vector<int> etaBin0;
	std::vector<int> numBins;
	std::vector<float> minEta;
	std::vector<float> delatEta;

	void reserve(int n) {
		isEndCap.reserve(n);
		etaBin0.reserve(n);
		numBins.reserve(n);
		minEta.reserve(n);
		deltaEta.reserve(n);
	}

	void addLayer(bool isEnd, int firstBin, int nBins, float mEta, float dEta) {
		isEndCap.push_back(isEnd);
		etaBin0.push_back(firstBin);
		numBins.push_back(nBins);
		minEta.push_back(mEta);
		deltaEta.push_back(dEta);
	}
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
	gbts_seedfinder_config() = delete;
	gbts_seedfinder_config(const std::vector<std::pair<int, int>>& binTables, const std::vector<device::gbts_layerInfo>& layerInfo,
              const std::vector<std::pair<uint64_t, int>>& detrayBarcodeBinning, float minPt);

	//layer linking and geometry	
	std::vector<std::pair<int, int>> binTables{};
	std::vector<device::gbts_layerInfo> layerInfo{};
	unsigned int nLayers = 0;	

	std::shared_ptr<int[]> volumeToLayerMap{};
	unsigned int maxVolIndex = 0;	

	std::vector<std::array<unsigned int, 2>> surfaceToLayerMap;
	unsigned int surfaceMapSize = 0;	

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

}
