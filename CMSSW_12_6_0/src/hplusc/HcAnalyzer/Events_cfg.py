import FWCore.ParameterSet.Config as cms

process = cms.Process("Hc")

process.load('Configuration.StandardSequences.Services_cff')
process.load('JetMETCorrections.Configuration.JetCorrectors_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('RecoTracker.Configuration.RecoTracker_cff')
#process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.GlobalTag.globaltag =  cms.string("106X_upgrade2018_realistic_v16_L1v1")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/HPlusCharm_3FS_MuRFScaleDynX0p50_HToZZTo4L_M125_TuneCP5_13TeV_amcatnlo_JHUGenV7011_pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2550000/50F51230-1751-6F43-84DA-C437A7F4783F.root")
)

process.mergedGenParticles = cms.EDProducer("MergedGenParticleProducer",
                                            inputPruned = cms.InputTag("prunedGenParticles"),
                                            inputPacked = cms.InputTag("packedGenParticles"),
                                            )


process.demo = cms.EDAnalyzer('HcAnalyzer',
packed = cms.InputTag("packedGenParticles"),
pruned = cms.InputTag("prunedGenParticles"),
merged = cms.InputTag("mergedGenParticles"),
tracks = cms.untracked.InputTag('packedPFCandidates'),
jets = cms.untracked.InputTag('slimmedJets'),
primaryVertices = cms.untracked.InputTag('offlineSlimmedPrimaryVertices'),
secVertices = cms.untracked.InputTag('slimmedSecondaryVertices'),
losttracks = cms.untracked.InputTag('lostTracks', '', "PAT"),
TrackPtCut = cms.untracked.double(0.5),
addPileupInfo = cms.untracked.InputTag('slimmedAddPileupInfo')
)

process.TFileService = cms.Service("TFileService",
        fileName = cms.string("hplusc_3FS_20k_0502.root"),
)

process.p = cms.Path(process.mergedGenParticles+process.demo)
