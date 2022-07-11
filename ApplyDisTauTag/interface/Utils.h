
enum class PFParticleType {
    Undefined = 0,  // undefined
    h = 1,          // charged hadron
    e = 2,          // electron
    mu = 3,         // muon
    gamma = 4,      // photon
    h0 = 5,         // neutral hadron
    h_HF = 6,       // HF tower identified as a hadron
    egamma_HF = 7,  // HF tower identified as an EM particle
};

inline PFParticleType TranslatePdgIdToPFParticleType(int pdgId)
{
    static const std::map<int, PFParticleType> type_map = {
        { 11, PFParticleType::e }, { 13, PFParticleType::mu }, { 22, PFParticleType::gamma },
        { 211, PFParticleType::h }, { 130, PFParticleType::h0 },
        { 1, PFParticleType::h_HF }, { 2, PFParticleType::egamma_HF },
    };
    auto iter = type_map.find(std::abs(pdgId));
    return iter == type_map.end() ? PFParticleType::Undefined : iter->second;
}