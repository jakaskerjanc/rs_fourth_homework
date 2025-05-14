#!/bin/sh



GEM5_WORKSPACE=/d/hpc/projects/FRI/GEM5/gem5_workspace
GEM5_ROOT=$GEM5_WORKSPACE/gem5
GEM5_PATH=$GEM5_ROOT/build/VEGA_X86/

APPTAINER_LOC=/d/hpc/projects/FRI/GEM5/gem5_workspace
APPTAINER_IMG=$APPTAINER_LOC/gcn-gpu_v24-0.sif

CUs=(2 4 8)
SIMD=(2 4 8)

# Loop over CUsß
for i in "${CUs[@]}"; do
    srun --ntasks=1 --time=00:30:00 --output=log_CU_${i}.txt  apptainer exec $APPTAINER_IMG $GEM5_PATH/gem5.opt --outdir=${1}_CU_${i}_stats $GEM5_ROOT/configs/example/apu_se.py -n 3 --num-compute-units ${i} --gfx-version="gfx902" -c ./histogram/bin/histogram_$1 &
done

# Loop over SIMD
for i in "${SIMD[@]}"; do
    srun --ntasks=1 --time=00:30:00 --output=log_SIMD_${i}.txt apptainer exec $APPTAINER_IMG $GEM5_PATH/gem5.opt --outdir=${1}_SIMD_${i}_stats $GEM5_ROOT/configs/example/apu_se.py -n 3 --simds-per-cu ${i} --gfx-version="gfx902" -c ./histogram/bin/histogram_$1 &
done

