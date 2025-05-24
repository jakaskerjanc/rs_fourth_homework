#!/bin/bash

# Metrics to extract from stats files
metrics=(
    "loadLatencyDist::mean"
    "vALUInsts"
    "groupReads"
    "groupWrites"
    "ldsBankAccess"
    "totalCycles"
    "vpc"
)

# Configurations
compute_units=(2 4 8)
implementations=("naive" "opt")

# Create CSV header
echo "Implementation,ComputeUnits,$(IFS=,; echo "${metrics[*]}")" > histogram_performance_results.csv

# Process each implementation and compute unit configuration
for impl in "${implementations[@]}"; do
    for cu in "${compute_units[@]}"; do
        # Path to the stats file
        stats_path="${impl}_CU_${cu}_stats/stats.txt"
        
        if [ ! -f "$stats_path" ]; then
            echo "Warning: $stats_path does not exist. Skipping..."
            continue
        fi
        
        echo -n "$impl,$cu" >> histogram_performance_results.csv
        
        # Extract metrics from stats file
        for metric in "${metrics[@]}"; do
            # Find the first occurrence of the metric (during GPU kernel execution)
            value=$(grep -m 1 "$metric" "$stats_path" | awk '{print $2}')
            
            if [ -z "$value" ]; then
                value="NA"
            fi
            
            echo -n ",$value" >> histogram_performance_results.csv
        done
        
        echo "" >> histogram_performance_results.csv
    done
done

echo "CSV file created: histogram_performance_results.csv"