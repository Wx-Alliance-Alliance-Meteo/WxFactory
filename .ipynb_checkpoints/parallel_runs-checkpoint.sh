#!/bin/bash

# Check if the correct number of arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 num_instances constant_value"
    exit 1
fi

num_instances=$1
constant_value=$2

for ((i = 1; i <= num_instances; i++)); do
    config_file="./config/colliding_inis/cold_bubble_${i}.ini"
    log_file="./random_logs/random_seed_log_${i}.txt"
    sys_iter_value=$((i * constant_value))
    
    # Copy the template config file to create the specific config file
    cp ./config/colliding_inis/template_cold_bubble.ini "$config_file"
    
    # Update the sys_iter value in the config file
    sed -i "s/^sys_iter = [0-9]\+/sys_iter = $sys_iter_value/" "$config_file"
    
    # Launch the datainit.sh script
    ./datainit.sh "$config_file" "$log_file" "$constant_value" &
    echo "Launched instance $i with config file $config_file and log file $log_file, sys_iter set to $sys_iter_value"
done

echo "All $num_instances instances launched."
