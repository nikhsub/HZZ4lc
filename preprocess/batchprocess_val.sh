#!/bin/bash

# Directory paths
#INPUT_DIR="/store/group/lpcljm/nvenkata/BTVH/toprocfiles/test"    # Set your input directory containing .root files
INPUT_DIR="/uscms/home/nvenkata/nobackup/higgs+c/preprocess/toproc_val"
OUTPUT_DIR="/uscms/home/nvenkata/nobackup/higgs+c/ML_scripts/files/validation/hplusc_val_1702"  # Set your output directory for .pkl files
#EOS_PREFIX="root://cmseos.fnal.gov/"
EOS_PREFIX=""

mkdir -p "$OUTPUT_DIR"

# Parameters for event processing
START_EVT=0
END_EVT=-1

# Loop through all ROOT files in the input directory
#for file_path in "$INPUT_DIR"/*.root; do
  # Extract file name without extension and set it as the save tag
#for file_path in $(xrdfsls "$INPUT_DIR" | grep '\.root$'); do
for file_path in "$INPUT_DIR"/*.root; do
  filename=$(basename "$file_path")
  save_tag="${filename%.root}"

  mod_file_path="${EOS_PREFIX}${file_path}"

  echo "$mod_file_path"

  # Run the processing script with the required arguments
  python process_val.py -d "$mod_file_path" -st "$save_tag" -s "$START_EVT" -e "$END_EVT"

  # Move the generated .pkl file to the output directory
  mv "evtvaldata_${save_tag}.pkl" "$OUTPUT_DIR/"
done

echo "Processing complete. All .pkl files have been moved to $OUTPUT_DIR."

