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
#include "traccc/utils/messaging.hpp"

//Detray include(s).
#include <detray/geometry/barcode.hpp>

namespace traccc::device {
//where to put these definitions?
struct gbts_layerInfo {
	std::vector<char> isEndcap;
	//etaBin0 and numBins
	std::vector<std::pair<int, int>> info;
	//minEta and deltaEta
	std::vector<std::pair<float, float>> geo;

	void reserve(int n) {
		isEndcap.reserve(n);
		info.reserve(n);
		geo.reserve(n);
	}

	void addLayer(bool isNotPixel, int firstBin, int nBins, float minEta, float etaBinWidth) {
		isEndcap.push_back(isNotPixel);
		info.push_back(std::make_pair(firstBin, nBins));
		geo.push_back(std::make_pair(minEta, etaBinWidth));
	}
};

struct gbts_consts {
	
	//CCA max iterations -> maxium seed length
	static constexpr short max_cca_iter = 20;
	//shared memory allocation sizes
	static constexpr short node_buffer_length = 250;
	static constexpr short shared_state_buffer_size = 608;
	
	// access into output graph
	static constexpr short node1 = 0;
	static constexpr short node2 = 1;
	static constexpr short nNei = 2;
	static constexpr short nei_start = 3;
};

}

namespace traccc {

struct gbts_algo_params {
	//edge making cuts
	float min_delta_phi = 0.015f;
	float dphi_coeff = 2.2e-4f;
	float min_delta_phi_low_dr = 0.002f;
	float dphi_coeff_low_dr = 4.33e-4f;	
	
	float minDeltaRadius = 2.0f;
	
	float min_z0 = -150.0f;
	float max_z0 = 150.0f;
	float maxOuterRadius = 550.0f;
	float cut_zMinU = min_z0 - maxOuterRadius*45;
	float cut_zMaxU = max_z0 + maxOuterRadius*45; //how to get ROI dzdr
	
	float max_Kappa = 3.75e-4f;
	float low_Kappa_d0 = 0.0f; //used to be 0.2f
	float high_Kappa_d0 = 0.0f; //used to be 1.0f

	//edge matching cuts
	float cut_dphi_max = 0.012f;
	float cut_dcurv_max = 0.001f;
	float cut_tau_ratio_max = 0.01f;
};

struct gbts_seedfinder_config {
    bool setLinkingScheme(const std::vector<std::pair<int, std::vector<int>>>& binTables, const device::gbts_layerInfo layerInfo,
    std::vector<std::pair<uint64_t, short>>& detrayBarcodeBinning, float minPt, std::unique_ptr<const ::Acts::Logger> logger);
	
	//layer linking and geometry	
	std::vector<std::pair<int, int>> binTables{};
	traccc::device::gbts_layerInfo layerInfo{};
	unsigned int nLayers  = 0;	

	std::vector<short> volumeToLayerMap{};
	std::vector<std::array<unsigned int, 2>> surfaceToLayerMap{};

	//tuned for 900 MeV pT cut and scaled by input minPt	
	gbts_algo_params algo_params{};	

	//node making bin counts
	int n_eta_bins = 0; //calculated from input layerInfo	
	unsigned int n_phi_bins = 120;
	//graph making maxiums
	unsigned char max_num_neighbours = 10;
	//graph extraction cuts
	int minLevel = 3; //equivlent to a cut of #seed edges or #spacepoints-1	
};

}
