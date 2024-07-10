#!/bin/bash

config_file="config/case2.ini"
n_processes=6
section="General"
key="sys_iter"

# Function to read the current run number
read_run_number() {
    grep -oP "(?<=${key} = )\d+" "$config_file"
}

# Function to update the run number
update_run_number() {
    local new_run_number=$1
    sed -i "s/${key} = [0-9]\+/${key} = ${new_run_number}/" "$config_file"
}

# Check if the section exists
if ! grep -q "\[$section\]" "$config_file"; then
    echo "Error: The section [$section] is not found in the configuration file."
    exit 1
fi

# Read the current run number
run_number=$(read_run_number)
if [ -z "$run_number" ]; then
    echo "Error: The key ${key} is not found in the section [$section]."
    exit 1
fi
echo "Current run number: $run_number"

# Increment the run number
new_run_number=$((run_number + 1))
echo "New run number: $new_run_number"

# Update the configuration file with the new run number
update_run_number $new_run_number

# Run the program using the Python interpreter
mpirun -n "$n_processes" python3 ./main_gef.py "$config_file"
