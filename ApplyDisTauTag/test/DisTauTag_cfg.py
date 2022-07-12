# coding: utf-8

import os

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing


# get the data/ directory
graph_file='/afs/desy.de/user/m/mykytaua/nfscms/softDeepTau/RecoML/DisTauTag/TauMLTools/Training/python/DisTauTag/mlruns/3/a27159734e304ea4b7f9e0042baa9e22/artifacts/model_graph/graph.pb'

# setup minimal options
options = VarParsing("python")
options.setDefault("inputFiles", "root://xrootd-cms.infn.it//store/mc/RunIIAutumn18MiniAOD/QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15_ext1-v2/270000/01825D94-A8FA-CB45-8048-7F6B6503DB45.root")  # noqa
options.parseArguments()

# define the process to run
process = cms.Process("TEST")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(10))
process.source = cms.Source("PoolSource",
    fileNames=cms.untracked.vstring(options.inputFiles))

# process options
process.options = cms.untracked.PSet(
    allowUnscheduled=cms.untracked.bool(True),
    wantSummary=cms.untracked.bool(True),
)

# setup DisTauTag by loading the auto-generated cfi (see DisTauTag.fillDescriptions)
process.load("TauMLTools.ApplyDisTauTag.disTauTag_cfi")
process.disTauTag.graphPath    = cms.string(graph_file)
process.disTauTag.jets         = cms.InputTag('slimmedJets')
process.disTauTag.pfCandidates = cms.InputTag('packedPFCandidates')

# define what to run in the path
process.p = cms.Path(process.disTauTag)