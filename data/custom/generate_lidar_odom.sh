#!/bin/bash

# Check if kiss-icp is installed
if command -v kiss_icp_pipeline >/dev/null 2>&1; then
    echo "kiss-icp is installed, proceeding..."
else
    pip install kiss-icp[all]
fi

# Define the base directory
BASE_DIR="sequences"

# Loop through all subdirectories in the BASE_DIR
for sequence_dir in "$BASE_DIR"/*/; do
    
    # Navigate to the lidar folder within the subsequence directory
    cd "$sequence_dir" || continue
    
    # Run the kiss_icp_pipeline command on the lidar folder
    echo "Running kiss-icp on $(pwd)"
    kiss_icp_pipeline lidar/
    
    # Find the data_poses.npy file in the 'results' subdirectory
    odom_pth=$(find results/*/lidar_poses.npy | head -1)
    
    # Move the found file to 'odom.npy' in the subsequence directory
    mv "$odom_pth" "lidar_odom.npy"
    
    # Remove the 'results' subdirectory and its contents
    rm -rf "results"
    
    # Return to the base directory
    cd ../..
done
