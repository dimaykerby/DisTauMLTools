namespace Scaling {
    struct PfCand{
        inline static const std::vector<std::vector<float>> mean = {{0,0},{50.0,50.0},{0.0,0.0},{0.0,0.0},{0.09469,0.09469},{0,0},{0,0},{0,0},{5.0,5.0},{5.0,5.0},{6.442,6.442},{0,0},{-0.01635,-0.01635},{0.02704,0.02704},{-0.01443,-0.01443},{0.05551,0.05551},{11.16,11.16},{13.53,13.53},{0.5,0.5},{0.5,0.5},{1.3,1.3},{0.5,0.5},{0,0},{0,0}};
        inline static const std::vector<std::vector<float>> std = {{1,1},{50.0,50.0},{3.0,3.0},{3.141592653589793,3.141592653589793},{0.0651,0.0651},{1,1},{1,1},{1,1},{5.0,5.0},{5.0,5.0},{8.344,8.344},{1,1},{2.4,2.4},{0.08913,0.08913},{7.444,7.444},{0.5998,0.5998},{60.39,60.39},{6.44,6.44},{0.5,0.5},{0.5,0.5},{1.3,1.3},{0.5,0.5},{1,1},{1,1}};
        inline static const std::vector<std::vector<float>> lim_min = {{-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()},{-1.0,-1.0},{-1.0,-1.0},{-1.0,-1.0},{-5,-5},{-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()},{-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()},{-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()},{-1.0,-1.0},{-1.0,-1.0},{-5,-5},{-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()},{-5,-5},{-5,-5},{-5,-5},{-5,-5},{-5,-5},{-5,-5},{-1.0,-1.0},{-1.0,-1.0},{-1.0,-1.0},{-1.0,-1.0},{-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()},{-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}};
        inline static const std::vector<std::vector<float>> lim_max = {{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()},{1.0,1.0},{1.0,1.0},{1.0,1.0},{5,5},{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()},{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()},{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()},{1.0,1.0},{1.0,1.0},{5,5},{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()},{5,5},{5,5},{5,5},{5,5},{5,5},{5,5},{1.0,1.0},{1.0,1.0},{1.0,1.0},{1.0,1.0},{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()},{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()}};
    };
    struct PfCandCategorical{
        inline static const std::vector<std::vector<float>> mean = {{0,0},{0,0},{0,0}};
        inline static const std::vector<std::vector<float>> std = {{1,1},{1,1},{1,1}};
        inline static const std::vector<std::vector<float>> lim_min = {{-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()},{-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()},{-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}};
        inline static const std::vector<std::vector<float>> lim_max = {{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()},{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()},{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()}};
    };
};

namespace Setup {
    const inline Bool_t debug = false;
    const inline long int n_tau = 500;
    const inline long int n_threads = 1;
    const inline std::string input_dir = "/nfs/dust/cms/user/mykytaua/softDeepTau/RecoML/DisTauTag/TauMLTools/FlatMerge-output-v3/";
    const inline std::string spectrum_to_reweight = "/nfs/dust/cms/user/mykytaua/softDeepTau/RecoML/DisTauTag/TauMLTools/FlatMerge-output-v3-spectrum/ShuffleMergeFlat.root";
    const inline std::string dataloader_core = "TauMLTools/Training/interface/DataLoaderDisTauTag_main.h";
    const inline std::unordered_map<int, std::string> jet_types_names = {{0,"jet"},{1,"tau"}};
    const inline long int output_classes = 2;
    const inline Bool_t recompute_jet_type = true;
    const inline long int weight_thr = 1000;
    const inline Bool_t to_propagate_glob = false;
    const inline std::vector<Double_t> yaxis = {20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,110.0,120.0,130.0,140.0,150.0,160.0,170.0,180.0,190.0,200.0,250.0,300.0,350.0,400.0,450.0,500.0,600.0,700.0,1000.0};
    const inline std::vector<std::vector<Double_t>> xaxis_list = {{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},{0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.5,1.8,2.1,2.4},{0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.5,1.8,2.1,2.4},{0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.5,1.8,2.1,2.4},{0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.5,1.8,2.1,2.4},{0.0,0.3,0.6,0.9,1.2,1.5,1.8,2.4},{0.0,0.3,0.6,0.9,1.2,1.5,1.8,2.4},{0.0,0.3,0.6,0.9,1.2,1.5,1.8,2.4}};
    const inline long int xmin = 0;
    const inline Double_t xmax = 2.4;
    const inline size_t n_Global = 5;
    const inline size_t n_PfCand = 24;
    const inline size_t nSeq_PfCand = 50;
    const inline size_t n_PfCandCategorical = 3;
    const inline size_t nSeq_PfCandCategorical = 50;
    const inline std::vector<std::string> CellObjectTypes {"PfCand","PfCandCategorical"};
};

enum class Global_Features {
    jet_pt = 0,
    jet_eta = 1,
    Lxy = 2,
    Lz = 3,
    Lrel = 4
};

enum class PfCand_Features {
    pfCand_valid = 0,
    pfCand_pt = 1,
    pfCand_eta = 2,
    pfCand_phi = 3,
    pfCand_mass = 4,
    pfCand_charge = 5,
    pfCand_puppiWeight = 6,
    pfCand_puppiWeightNoLep = 7,
    pfCand_lostInnerHits = 8,
    pfCand_nPixelHits = 9,
    pfCand_nHits = 10,
    pfCand_hasTrackDetails = 11,
    pfCand_dxy = 12,
    pfCand_dxy_error = 13,
    pfCand_dz = 14,
    pfCand_dz_error = 15,
    pfCand_track_chi2 = 16,
    pfCand_track_ndof = 17,
    pfCand_caloFraction = 18,
    pfCand_hcalFraction = 19,
    pfCand_rawCaloFraction = 20,
    pfCand_rawHcalFraction = 21,
    pfCand_deta = 22,
    pfCand_dphi = 23
};

enum class PfCandCategorical_Features {
    pfCand_particleType = 0,
    pfCand_pvAssociationQuality = 1,
    pfCand_fromPV = 2
};

enum class CellObjectType {
    PfCand,
    PfCandCategorical
};

template<typename T> struct FeaturesHelper;

template<> struct FeaturesHelper<PfCand_Features> {
    static constexpr CellObjectType object_type = CellObjectType::PfCand;
    static constexpr size_t size = 24;
    static constexpr size_t length = 50;
    using scaler_type = Scaling::PfCand;
};

template<> struct FeaturesHelper<PfCandCategorical_Features> {
    static constexpr CellObjectType object_type = CellObjectType::PfCandCategorical;
    static constexpr size_t size = 3;
    static constexpr size_t length = 50;
    using scaler_type = Scaling::PfCandCategorical;
};

using FeatureTuple = std::tuple<PfCand_Features, PfCandCategorical_Features>;