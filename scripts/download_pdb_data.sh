#!/bin/bash
# download_pdb_data.sh: This script creates a temporary virtual environment,
# installs gdown, downloads the dataset using the URL specified in the .env file (located in the parent directory of the script),
# optionally unzips the downloaded file if it is a ZIP archive (based on the UNZIP_FILE environment variable),
# conditionally deletes the ZIP file if REMOVE_ZIP is set to true,
# and then removes the temporary virtual environment.
#
# If the virtual environment cannot be created because python3-venv is missing,
# the script will attempt to install it automatically using apt-get.
#
# A trap is set to ensure that the temporary virtual environment is removed even if the script is interrupted.

# Exit immediately if any command fails
set -e

# Determine the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# The .env file is expected to be in the parent directory of the script
ENV_FILE="$SCRIPT_DIR/../.env"

# Define the temporary virtual environment directory (relative to the current directory)
TMP_VENV="$SCRIPT_DIR/tmp_venv"

# Define a cleanup function to remove the temporary virtual environment.
cleanup() {
    echo "Cleaning up temporary virtual environment..."
    # Deactivate the virtual environment if it is active.
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    # Remove the temporary virtual environment directory if it exists.
    if [ -d "$TMP_VENV" ]; then
        rm -rf "$TMP_VENV"
    fi
}

# Set a trap to call cleanup on script exit (whether successful or due to interruption)
trap cleanup EXIT

echo "Creating temporary virtual environment in $TMP_VENV..."

# Function to create the virtual environment
create_venv() {
    python3 -m venv "$TMP_VENV"
}

# Try to create the virtual environment; if it fails, attempt to install python3-venv and retry.
if ! create_venv; then
    echo "Error: The virtual environment was not created successfully."
    if command -v apt-get >/dev/null 2>&1; then
        echo "Attempting to install python3-venv..."
        sudo apt-get update && sudo apt-get install -y python3-venv
        echo "Re-trying virtual environment creation..."
        if ! create_venv; then
            echo "Error: Failed to create the virtual environment even after installing python3-venv."
            exit 1
        fi
    else
        echo "Automatic installation of python3-venv is not supported on this system."
        exit 1
    fi
fi

echo "Activating temporary virtual environment..."
source "$TMP_VENV/bin/activate"

echo "Upgrading pip and installing gdown..."
pip install --upgrade pip
pip install gdown

# Check if the .env file exists using the calculated path.
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found in the parent directory: $ENV_FILE"
    exit 1
fi

# Load environment variables from the .env file
set -a
source "$ENV_FILE"
set +a

# Debug: print loaded environment variables
echo "Environment variables loaded:"
echo "  PDB_DATA_URL: $PDB_DATA_URL"
echo "  PDB_DATA_OUTPUT_DIR: $PDB_DATA_OUTPUT_DIR"
echo "  PDB_DATA_ZIP_NAME (raw): $PDB_DATA_ZIP_NAME"
echo "  UNZIP_FILE: $UNZIP_FILE"
echo "  REMOVE_ZIP: $REMOVE_ZIP"

# Check that PDB_DATA_URL is defined
if [ -z "$PDB_DATA_URL" ]; then
    echo "Error: PDB_DATA_URL is not set in the .env file."
    exit 1
fi

# Set default output directory if PDB_DATA_OUTPUT_DIR is not defined
if [ -z "$PDB_DATA_OUTPUT_DIR" ]; then
    PDB_DATA_OUTPUT_DIR="data/pdb"
fi

# Ensure the output directory exists
mkdir -p "$PDB_DATA_OUTPUT_DIR"

# Use a default output filename if PDB_DATA_ZIP_NAME is not specified.
# If provided as a filename only (without a slash), prepend the output directory.
if [ -z "$PDB_DATA_ZIP_NAME" ]; then
    PDB_DATA_ZIP_NAME="$PDB_DATA_OUTPUT_DIR/dataset.zip"
else
    if [[ "$PDB_DATA_ZIP_NAME" != */* ]]; then
        PDB_DATA_ZIP_NAME="$PDB_DATA_OUTPUT_DIR/$PDB_DATA_ZIP_NAME"
    fi
fi

echo "Final zip file will be saved as: $PDB_DATA_ZIP_NAME"
echo "Downloading dataset from: $PDB_DATA_URL"
echo "Saving as: $PDB_DATA_ZIP_NAME"

# Download the dataset using gdown with the --fuzzy flag for flexible URL parsing
gdown --fuzzy "$PDB_DATA_URL" --output "$PDB_DATA_ZIP_NAME"

echo "Dataset downloaded successfully."

# if the ZIP archive downloaded successfully, optionally unzip it and conditionally remove the ZIP file.
if [[ "${UNZIP_FILE,,}" == "true" ]]; then
    echo "UNZIP_FILE is set to true. Unzipping dataset into $PDB_DATA_OUTPUT_DIR..."
    if ! command -v unzip >/dev/null 2>&1; then
        echo "unzip command not found. Attempting to install unzip..."
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y unzip
        else
            echo "Automatic installation of unzip is not supported on this system."
            exit 1
        fi
    fi
    unzip -o "$PDB_DATA_ZIP_NAME" -d "$PDB_DATA_OUTPUT_DIR"
    echo "Dataset unzipped successfully."
else
    echo "UNZIP_FILE is not set to true. Skipping unzipping process."
fi

if [[ "${REMOVE_ZIP,,}" == "true" ]]; then
    echo "REMOVE_ZIP is set to true. Deleting the ZIP file: $PDB_DATA_ZIP_NAME"
    rm -f "$PDB_DATA_ZIP_NAME"
else
    echo "REMOVE_ZIP is not set to true. Keeping the ZIP file."
fi

echo "All tasks completed successfully."