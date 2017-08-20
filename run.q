#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=70:00:00
#PBS -l mem=32GB
#PBS -N Final_test_HIST_Continue_Fixed
#PBS -M sc6170@nyu.edu
#PBS -j oe

module purge
module load tensorflow/python2.7/20161029
module load scikit-image/intel/0.11.3
module load scipy/intel/0.18.0
module load h5py/intel/2.6.0


RUNDIR=$SCRATCH/IVP/root/Project
cd $RUNDIR/Code-7_8G

python avg_runner.py -r 24 -n $PBS_JOBNAME
#python avg_runner.py -r 24 -l '../../Save/Dropv2_LAP_TEACH_RES_FeatureInject_Lamb1_WGAN_CITER1_NOBN@2017-03-10_09_48_40/Models/model.ckpt-125000' -T
#python avg_runner.py -r 24 -l '../../Save/Dropv2_LAP_TEACH_FEATv2_FeatureInject_Lamb1_GAN_BN@2017-03-05_10_27_14/Models/model.ckpt-20000' -A
# leave a blank line at the end
