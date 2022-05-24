/*! Create histograms with pt and eta distribution for every type of tau (tau_h, tau_mu, tau_e, tau_j)
*/
#include "TauMLTools/Core/interface/program_main.h"
#include "TauMLTools/Analysis/interface/TauTuple.h"
#include "TauMLTools/Analysis/interface/SummaryTuple.h"
#include "TauMLTools/Core/interface/RootFilesMerger.h"
#include "TauMLTools/Core/interface/NumericPrimitives.h"

#include "TauMLTools/Core/interface/AnalyzerData.h"
#include "TauMLTools/Core/interface/RootExt.h"
#include "TauMLTools/Analysis/interface/TauSelection.h"
#include "TauMLTools/Analysis/interface/TauTagSelection.h"

#include <iostream>
#include <fstream>

struct Arguments {
    run::Argument<std::string> outputfile{"outputfile", "output file name"};
    run::Argument<std::string> output_entries{"output_entries", "txt output file with filenames and number of entries"};
    run::Argument<std::vector<std::string>> input_dirs{"input-dir", "input directory"};
    run::Argument<std::string> pt_hist{"pt-hist", "pt hist setup: (number of bins, pt_min, pt_max)", "200, 0.0, 1000"};
    run::Argument<std::string> eta_hist{"eta-hist", "eta hist setup: (number of bins, abs(eta)_min, abs(eta)_max)","4, 0.0, 2.3"};
    run::Argument<std::string> file_name_pattern{"file-name-pattern", "regex expression to match file names",
                                                 "^.*\\.root$"};
    run::Argument<std::string> exclude_list{"exclude-list", "comma separated list of files to exclude", ""};
    run::Argument<std::string> exclude_dir_list{"exclude-dir-list","comma separated list of directories to exclude", ""};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
    run::Argument<int> n_files{"n-files", "number of files to process (for all: -1)", -1};
};

struct HistArgs {
  int eta_bins, pt_bins;
  double eta_min, eta_max, pt_min, pt_max;
  HistArgs(const std::vector<std::string>& args_pt, const std::vector<std::string>& args_eta)
  {
    pt_bins = analysis::Parse<int>(args_pt.at(0));
    pt_min = analysis::Parse<double>(args_pt.at(1));
    pt_max = analysis::Parse<double>(args_pt.at(2));
    eta_bins = analysis::Parse<int>(args_eta.at(0));
    eta_min = analysis::Parse<double>(args_eta.at(1));
    eta_max = analysis::Parse<double>(args_eta.at(2));
  }
};

class HistSpectrum : public root_ext::AnalyzerData {

public:
  HistSpectrum(std::shared_ptr<TFile> outputFile, const HistArgs& hargs) : root_ext::AnalyzerData(outputFile)
  {
    jet_eta_pt.SetMasterHist(hargs.eta_bins, hargs.eta_min, hargs.eta_max,
                             hargs.pt_bins, hargs.pt_min, hargs.pt_max);
  }

  TH2D_ENTRY(jet_eta_pt, 4, 0, 2.3, 200, 0, 1000)
  TH1D_ENTRY(jet_valid, 3, -1, 2)

};

namespace analysis {
class TargetedSamplingHists {

public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;

    TargetedSamplingHists(const Arguments& args) :
      input_files(RootFilesMerger::FindInputFiles(args.input_dirs(),
                                                  args.file_name_pattern(),
                                                  args.exclude_list(),
                                                  args.exclude_dir_list())),
      outputfile(root_ext::CreateRootFile(args.outputfile()))
    {
      output_txt.open(args.output_entries(), std::ios::trunc);

      ROOT::EnableThreadSafety();
      if(args.n_threads() > 1) ROOT::EnableImplicitMT(args.n_threads());

      auto par_path = GetPathWithoutFileName(args.outputfile());
      if(!boost::filesystem::exists(par_path)) boost::filesystem::create_directory(par_path);

      hists = std::make_shared<HistSpectrum>(outputfile,
             ParseHistSetup(args.pt_hist(),args.eta_hist()));

      if(args.n_files() > 0) input_files.resize(args.n_files());
    }

    void Run()
    {
        for(const auto& file_name : input_files) {
            std::cout << "file: " << file_name << std::endl;
            auto file = root_ext::OpenRootFile(file_name);

            const std::set<std::string> enabled_branches = {
                  "jet_index", "jet_pt", "jet_eta",
                  "genJet_pt", "genjet_pt",
                  "genLepton_kind",
                  "genLepton_lastMotherIndex",
                  "genParticle_pdgId",
                  "genParticle_mother",
                  "genParticle_charge",
                  "genParticle_isFirstCopy",
                  "genParticle_isLastCopy",
                  "genParticle_pt",
                  "genParticle_eta",
                  "genParticle_phi",
                  "genParticle_mass",
                  "genParticle_vtx_x",
                  "genParticle_vtx_y",
                  "genParticle_vtx_z",
                  "evt"
              };

            auto tauTuple = TauTuple("taus", file.get(), true, {}, enabled_branches);

            output_txt << file_name << " " << tauTuple.GetEntries() << "\n";

            for(const Tau& tau : tauTuple)
            {
              AddJetCandidate(tau);
              ++total_size;
            }
        }

        std::cout << "All file has been processed." << std::endl
                  << "Number of files = " << input_files.size() << std::endl
                  << "Number of processed taus = " << total_size << std::endl
                  << "Number of tau-jets = " << static_cast<Int_t>(hists->jet_eta_pt("tau").Integral()) << std::endl
                  << "Number of bkgr-jets = " << static_cast<Int_t>(hists->jet_eta_pt("jet").Integral()) << std::endl
                  << "Number of not valid taus = " << static_cast<Int_t>(hists->jet_valid().GetBinContent(2)) << std::endl
                  << "Number of valid taus = " << static_cast<Int_t>(hists->jet_valid().GetBinContent(3)) << std::endl;
        
        output_txt.close();
    }

private:
    static HistArgs ParseHistSetup(const std::string& pt_hist, const std::string& eta_hist)
    {
        const auto& split_args_pt = SplitValueList(pt_hist, true, ",", true);
        const auto& split_args_eta = SplitValueList(eta_hist, true, ",", true);

        std::cout << "pt histogram setup (n_bins pt_min pt_max): ";
        for(const std::string& bin_str : split_args_pt) std::cout << Parse<double>(bin_str) << "  ";
        std::cout << std::endl;

        std::cout << "eta histogram setup (n_bins eta_min eta_max): ";
        for(const std::string& bin_str : split_args_eta) std::cout << Parse<double>(bin_str) << "  ";
        std::cout << std::endl;

        if(split_args_pt.size()!=3 || Parse<double>(split_args_pt[0])<1 || Parse<double>(split_args_pt[1])>=Parse<double>(split_args_pt[2]))
        throw exception("Invalid pt-hist arguments");

        if(split_args_eta.size()!=3 || Parse<double>(split_args_eta[0])<1 || Parse<double>(split_args_eta[1])>=Parse<double>(split_args_eta[2]))
        throw exception("Invalid eta-hist arguments");

        HistArgs histarg(split_args_pt, split_args_eta);

        return histarg;
    }

    void AddJetCandidate(const Tau& tau)
    {
        const auto JetType_match = GetJetType(tau);

        if(JetType_match != boost::none)
        {
            hists->jet_valid().Fill(1);
            if( JetType_match == JetType::Tau)
                hists->jet_eta_pt("tau").Fill(std::abs(tau.jet_eta), tau.jet_pt);
            else if( JetType_match == JetType::Jet)
                hists->jet_eta_pt("jet").Fill(std::abs(tau.jet_eta), tau.jet_pt);
            else
                throw exception("Error AddJetCandidate: unknown jet type.");
        } else {
            hists->jet_valid().Fill(0);
        }

    }

    bool PassSelection(const Tau& tau) const
    {
      return (tau.tau_index >= 0);
    }

  private:
      std::vector<std::string> input_files;
      std::shared_ptr<TFile> outputfile;
      std::ofstream output_txt;
      std::shared_ptr<HistSpectrum> hists;
      Int_t total_size=0;

};

} // namespace analysis

PROGRAM_MAIN(analysis::TargetedSamplingHists, Arguments)
