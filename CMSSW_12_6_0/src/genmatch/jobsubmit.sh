#!/bin/sh


INPATH=/eos/uscms$(xrdfsls /store/group/lpcljm/nvenkata/Hplusc/hplusc_1402/HPlusCharm_3FS_MuRFScaleDynX0p50_HToZZTo4L_M125_TuneCP5_13TeV_amcatnlo_JHUGenV7011_pythia8/MC_hplusc3FS_1402)/0000

#INPATH=/eos/uscms/store/group/lpcljm/nvenkata/BTVH/files


python condor_analyze.py -i $INPATH -o 1702_hplusc_3FS -ec 3000
