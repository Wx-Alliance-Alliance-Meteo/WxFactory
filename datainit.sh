#!/bin/bash

# Check if the correct number of arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 config_file num_runs"
    exit 1
fi

config_file=$1
num_runs=$2  # Number of times to run the program

n_processes=6
section="General"
key1="sys_iter"
key2="rand_seed"
min_seed=1000  # Minimum seed value
max_seed=999999  # Maximum seed value

log_file="./random_logs.txt"

# Function to read the current value of a key
read_key_value() {
    local key=$1
    grep -oP "(?<=${key} = )\d+" "$config_file"
}

# Function to update the value of a key
update_key_value() {
    local key=$1
    local new_value=$2
    sed -i "s/${key} = [0-9]\+/${key} = ${new_value}/" "$config_file"
}

# Check if the section exists
if ! grep -q "\[$section\]" "$config_file"; then
    echo "Error: The section [$section] is not found in the configuration file."
    exit 1
fi

# Loop to run the program multiple times
for ((i = 1; i <= num_runs; i++)); do
    echo "====================="
    echo "Starting run number $i"
    
    # Read the current value of key1 (sys_iter)
    sys_iter=$(read_key_value "$key1")
    if [ -z "$sys_iter" ]; then
        echo "Error: The key ${key1} is not found in the section [$section]."
        exit 1
    fi
    echo "Current $key1: $sys_iter"
    
    # Generate a new random seed in the specified range
    rand_seed=$((RANDOM % (max_seed - min_seed + 1) + min_seed))
    echo "Generated new $key2: $rand_seed"
    
    # Increment the sys_iter value
    new_sys_iter=$((sys_iter + 1))
    echo "New $key1: $new_sys_iter"
    
    # Update the configuration file with the new sys_iter and rand_seed values
    update_key_value "$key1" "$new_sys_iter"
    update_key_value "$key2" "$rand_seed"
    
    # Log the new random seed to the log file
    echo "Run $sys_iter: $rand_seed" >> "$log_file"
    
    # Run the program using the Python interpreter
    ./main_gef.py "$config_file"
    
    echo "Run number $i completed"
    echo "====================="
done

echo "All runs completed successfully"

