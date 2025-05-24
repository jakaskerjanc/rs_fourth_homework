#!/bin/sh

GEM5_WORKSPACE=/d/hpc/projects/FRI/GEM5/gem5_workspace
GEM5_ROOT=$GEM5_WORKSPACE/gem5
GEM5_PATH=$GEM5_ROOT/build/VEGA_X86/

APPTAINER_LOC=/d/hpc/projects/FRI/GEM5/gem5_workspace
APPTAINER_IMG=$APPTAINER_LOC/gcn-gpu_v24-0.sif

# First build both histogram implementations
# cd histogram && ./make_apptainer.sh && cd ..

# Run naive histogram with 2, 4, and 8 compute units
srun --ntasks=1 --time=00:30:00 --output=log_naive_CU2.txt apptainer exec $APPTAINER_IMG $GEM5_PATH/gem5.opt --outdir=naive_CU_2_stats $GEM5_ROOT/configs/example/apu_se.py -n 3 --num-compute-units 2 --gfx-version="gfx902" -c ./histogram/bin/histogram_naive

srun --ntasks=1 --time=00:30:00 --output=log_naive_CU4.txt apptainer exec $APPTAINER_IMG $GEM5_PATH/gem5.opt --outdir=naive_CU_4_stats $GEM5_ROOT/configs/example/apu_se.py -n 3 --num-compute-units 4 --gfx-version="gfx902" -c ./histogram/bin/histogram_naive

srun --ntasks=1 --time=00:30:00 --output=log_naive_CU8.txt apptainer exec $APPTAINER_IMG $GEM5_PATH/gem5.opt --outdir=naive_CU_8_stats $GEM5_ROOT/configs/example/apu_se.py -n 3 --num-compute-units 8 --gfx-version="gfx902" -c ./histogram/bin/histogram_naive

# Run optimized histogram with 2, 4, and 8 compute units
srun --ntasks=1 --time=02:00:00 --output=log_opt_CU2.txt apptainer exec $APPTAINER_IMG $GEM5_PATH/gem5.opt --outdir=opt_CU_2_stats $GEM5_ROOT/configs/example/apu_se.py -n 3 --num-compute-units 2 --gfx-version="gfx902" -c ./histogram/bin/histogram_opt

srun --ntasks=1 --time=00:30:00 --output=log_opt_CU4.txt apptainer exec $APPTAINER_IMG $GEM5_PATH/gem5.opt --outdir=opt_CU_4_stats $GEM5_ROOT/configs/example/apu_se.py -n 3 --num-compute-units 4 --gfx-version="gfx902" -c ./histogram/bin/histogram_opt

srun --ntasks=1  --time=02:00:00 --output=log_opt_CU8.txt apptainer exec $APPTAINER_IMG $GEM5_PATH/gem5.opt --outdir=opt_CU_8_stats $GEM5_ROOT/configs/example/apu_se.py -n 3 --num-compute-units 8 --gfx-version="gfx902" -c ./histogram/bin/histogram_opt
 


