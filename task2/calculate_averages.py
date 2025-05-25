#!/usr/bin/env python3
import re
import csv
from collections import defaultdict

def parse_test_output(filename):
    """Parse the test output file and extract timing data."""
    data = defaultdict(lambda: {'tensorcore': [], 'without_tensorcore': []})
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find all matrix size and timing entries
    size_pattern = r'Running matrix multiplication for size (\d+) x (\d+)'
    tensorcore_pattern = r'Time taken with TensorCore: ([\d.]+) ms'
    without_tensorcore_pattern = r'Time taken without TensorCore: ([\d.]+) ms'
    
    lines = content.split('\n')
    current_size = None
    
    for line in lines:
        size_match = re.search(size_pattern, line)
        if size_match:
            current_size = int(size_match.group(1))
            continue
            
        if current_size is not None:
            tensorcore_match = re.search(tensorcore_pattern, line)
            if tensorcore_match:
                data[current_size]['tensorcore'].append(float(tensorcore_match.group(1)))
                continue
                
            without_tensorcore_match = re.search(without_tensorcore_pattern, line)
            if without_tensorcore_match:
                data[current_size]['without_tensorcore'].append(float(without_tensorcore_match.group(1)))
                continue
    
    return data

def calculate_averages(data):
    """Calculate averages for each matrix size."""
    averages = {}
    
    for size, timings in data.items():
        tensorcore_avg = sum(timings['tensorcore']) / len(timings['tensorcore'])
        without_tensorcore_avg = sum(timings['without_tensorcore']) / len(timings['without_tensorcore'])
        speedup = without_tensorcore_avg / tensorcore_avg
        
        averages[size] = {
            'tensorcore_avg': tensorcore_avg,
            'without_tensorcore_avg': without_tensorcore_avg,
            'speedup': speedup,
            'num_runs': len(timings['tensorcore'])
        }
    
    return averages

def write_csv(averages, filename):
    """Write the averages to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Matrix Size', 'TensorCore Avg (ms)', 'Without TensorCore Avg (ms)', 'Speedup', 'Number of Runs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for size in sorted(averages.keys()):
            avg_data = averages[size]
            writer.writerow({
                'Matrix Size': f'{size}x{size}',
                'TensorCore Avg (ms)': f'{avg_data["tensorcore_avg"]:.6f}',
                'Without TensorCore Avg (ms)': f'{avg_data["without_tensorcore_avg"]:.6f}',
                'Speedup': f'{avg_data["speedup"]:.2f}x',
                'Number of Runs': avg_data["num_runs"]
            })

def main():
    # Parse the test output
    data = parse_test_output('test_tc.out')
    
    # Calculate averages
    averages = calculate_averages(data)
    
    # Print summary to console
    print("Matrix Multiplication Performance Summary:")
    print("=" * 60)
    for size in sorted(averages.keys()):
        avg_data = averages[size]
        print(f"Matrix Size: {size}x{size}")
        print(f"  TensorCore Average: {avg_data['tensorcore_avg']:.6f} ms")
        print(f"  Without TensorCore Average: {avg_data['without_tensorcore_avg']:.6f} ms")
        print(f"  Speedup: {avg_data['speedup']:.2f}x")
        print(f"  Number of runs: {avg_data['num_runs']}")
        print()
    
    # Write to CSV
    write_csv(averages, 'matrix_multiplication_averages.csv')
    print("Results saved to 'matrix_multiplication_averages.csv'")

if __name__ == "__main__":
    main() 