/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/utils/helpers.hpp"

namespace traccc {

namespace plot_helpers {

binning::binning() {}

binning::binning(std::string b_title, int bins, float b_min, float b_max)
    : title(b_title), n_bins(bins), min(b_min), max(b_max) {}

TH1F* book_histo(const char* hist_name, const char* hist_title,
                 const binning& var_binning) {
    TH1F* hist = new TH1F(hist_name, hist_title, var_binning.n_bins,
                          var_binning.min, var_binning.max);
    hist->GetXaxis()->SetTitle(var_binning.title.c_str());
    hist->GetYaxis()->SetTitle("Entries");
    hist->Sumw2();
    return hist;
}

void fill_histo(TH1F* hist, float value, float weight) {
    assert(hist != nullptr);
    hist->Fill(value, weight);
}

void ana_histo(TH1D* input_hist, int j, TH1F* mean_hist, TH1F* width_hist) {
    // evaluate mean and width via the Gauss fit
    assert(input_hist != nullptr);
    if (input_hist->GetEntries() > 0) {
        TFitResultPtr r = input_hist->Fit("gaus", "QS0");
        if (r.Get() and ((r->Status() % 1000) == 0)) {
            // fill the mean and width into 'j'th bin of the meanHist and
            // widthHist, respectively
            mean_hist->SetBinContent(j, r->Parameter(1));
            mean_hist->SetBinError(j, r->ParError(1));
            width_hist->SetBinContent(j, r->Parameter(2));
            width_hist->SetBinError(j, r->ParError(2));
        }
    }
}

TEfficiency* book_eff(const char* eff_name, const char* eff_title,
                      const binning& var_binning) {
    TEfficiency* efficiency =
        new TEfficiency(eff_name, eff_title, var_binning.n_bins,
                        var_binning.min, var_binning.max);
    return efficiency;
}

void fill_eff(TEfficiency* efficiency, float value, bool status) {
    assert(efficiency != nullptr);
    efficiency->Fill(status, value);
}

TProfile* book_prof(const char* prof_name, const char* prof_title,
                    const binning& var_x_binning,
                    const binning& var_y_binning) {
    TProfile* prof = new TProfile(prof_name, prof_title, var_x_binning.n_bins,
                                  var_x_binning.min, var_x_binning.max,
                                  var_y_binning.min, var_y_binning.max);
    prof->GetXaxis()->SetTitle(var_x_binning.title.c_str());
    prof->GetYaxis()->SetTitle(var_y_binning.title.c_str());
    return prof;
}

void fill_prof(TProfile* profile, float x_value, float y_value, float weight) {
    assert(profile != nullptr);
    profile->Fill(x_value, y_value, weight);
}

}  // namespace plot_helpers

}  // namespace traccc
