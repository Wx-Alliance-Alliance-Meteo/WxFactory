#!/bin/bash

# Function to update sys_iter in a config file
update_sys_iter() {
    local file=$1
    local value=$2
    # Use awk to update the sys_iter value in the General section
    awk -v new_val="$value" '
    BEGIN {FS=OFS=" = "}
    /^\[General\]/ {in_general=1; print; next}
    in_general && /^sys_iter/ {$2=new_val; in_general=0}
    {print}
    ' "$file" > tmpfile && mv tmpfile "$file"
}

# Check if correct number of arguments are passed
if [ $# -ne 3 ]; then
    echo "Usage: $0 <path_to_config_folder> <num_runs> <start_count>"
    exit 1
fi

config_folder=$1
num_runs=$2
start_count=$3

# Check if num_runs is a multiple of 12
if (( num_runs % 12 != 0 )); then
    echo "Error: num_runs must be a multiple of 12."
    exit 1
fi

# Calculate the increment value
increment=$((num_runs / 12))

# Update sys_iter in each config file
for i in $(seq 1 12); do
    config_file="$config_folder/cold_bubble_$i.ini"
    sys_iter_value=$((start_count + increment * (i-1)))
    if [ -f "$config_file" ]; then
        update_sys_iter "$config_file" "$sys_iter_value"
        echo "Updated $config_file: sys_iter set to $sys_iter_value"
    else
        echo "Config file $config_file not found!"
    fi
done


