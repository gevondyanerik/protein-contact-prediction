#!/usr/bin/env bash

# ==============================================================================================
# This script sets up a dataset download pipeline by:
#   1. Loading environment variables from a .env file.
#   2. Creating and managing a temporary Python virtual environment.
#   3. Installing necessary dependencies.
#   4. Downloading a dataset from a given URL using gdown.
#   5. Optionally unzipping the dataset if required.
#   6. Cleaning up the virtual environment upon script exit.
#
# Usage:
#   - Ensure that a .env file exists in the parent directory with the required variables.
#   - Run the script: `./dataset_setup.sh`
#
# Environment Variables (Loaded from ../.env):
#   - PDB_DATA_URL: The URL from which the dataset is downloaded.
#   - PDB_DATA_OUTPUT_DIR: Directory where the dataset will be stored.
#   - PDB_DATA_ZIP_NAME: Name of the downloaded ZIP file.
#   - UNZIP_FILE: Whether to unzip the downloaded dataset (true/false).
#   - REMOVE_ZIP: Whether to delete the ZIP file after extraction (true/false).
#
# Notes:
#   - The script exits immediately if any command fails (`set -e`).
#   - If python3-venv is missing, it attempts to install it.
#   - The script creates a temporary virtual environment that is deleted upon exit.
# ==============================================================================================

# Load environment variables from the .env file located in the parent directory.
source ../.env

# Exit immediately if any command exits with a non-zero status.
set -e

# Determine the directory in which this script resides.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Define the path to the .env file (in the parent directory).
ENV_FILE="$SCRIPT_DIR/../.env"
# Define a temporary virtual environment directory.
TMP_VENV="$SCRIPT_DIR/tmp_venv"

# Define a cleanup function to deactivate and remove the temporary virtual environment.
cleanup() {
    echo "Cleaning up temporary virtual environment..."
    # Deactivate the virtual environment if it's active.
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    # Remove the temporary virtual environment directory if it exists.
    if [ -d "$TMP_VENV" ]; then
        rm -rf "$TMP_VENV"
    fi
}

# Set a trap so that the cleanup function is called on script exit, regardless of success or error.
trap cleanup EXIT

echo "Creating temporary virtual environment in $TMP_VENV..."

# Function to create a virtual environment.
create_venv() {
    python3 -m venv "$TMP_VENV"
}

# Try to create the virtual environment. If creation fails, attempt to install python3-venv and retry.
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
# Activate the newly created virtual environment.
source "$TMP_VENV/bin/activate"

echo "Upgrading pip and installing gdown..."
# Upgrade pip and install gdown using pip.
pip install --upgrade pip
pip install gdown

# Check if the .env file exists using the calculated path.
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found in the parent directory: $ENV_FILE"
    exit 1
fi

# Export all variables from the .env file to the environment.
set -a
source "$ENV_FILE"
set +a

# Print the loaded environment variables for debugging.
echo "Environment variables loaded:"
echo "  PDB_DATA_URL: $PDB_DATA_URL"
echo "  PDB_DATA_OUTPUT_DIR: $PDB_DATA_OUTPUT_DIR"
echo "  PDB_DATA_ZIP_NAME: $PDB_DATA_ZIP_NAME"
echo "  UNZIP_FILE: $UNZIP_FILE"
echo "  REMOVE_ZIP: $REMOVE_ZIP"

# Ensure that PDB_DATA_URL is set.
if [ -z "$PDB_DATA_URL" ]; then
    echo "Error: PDB_DATA_URL is not set in the .env file."
    exit 1
fi

# If PDB_DATA_OUTPUT_DIR is not set, use the current value.
if [ -z "$PDB_DATA_OUTPUT_DIR" ]; then
    PDB_DATA_OUTPUT_DIR=$PDB_DATA_OUTPUT_DIR
fi

# Create the output directory if it does not exist.
mkdir -p "$PDB_DATA_OUTPUT_DIR"

# Set a default zip file name if not provided.
if [ -z "$PDB_DATA_ZIP_NAME" ]; then
    PDB_DATA_ZIP_NAME="$PDB_DATA_OUTPUT_DIR/dataset.zip"
else
    # If only a filename is given, prepend the output directory.
    if [[ "$PDB_DATA_ZIP_NAME" != */* ]]; then
        PDB_DATA_ZIP_NAME="$PDB_DATA_OUTPUT_DIR/$PDB_DATA_ZIP_NAME"
    fi
fi

echo "Final zip file will be saved as: $PDB_DATA_ZIP_NAME"
echo "Downloading dataset from: $PDB_DATA_URL"
echo "Saving as: $PDB_DATA_ZIP_NAME"

# Download the dataset using gdown with --fuzzy flag for flexible URL parsing.
gdown --fuzzy "$PDB_DATA_URL" --output "$PDB_DATA_ZIP_NAME"

echo "Dataset downloaded successfully."

# If UNZIP_FILE is set to true, unzip the downloaded file into the output directory.
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

# If REMOVE_ZIP is set to true, delete the downloaded zip file.
if [[ "${REMOVE_ZIP,,}" == "true" ]]; then
    echo "REMOVE_ZIP is set to true. Deleting the ZIP file: $PDB_DATA_ZIP_NAME"
    rm -f "$PDB_DATA_ZIP_NAME"
else
    echo "REMOVE_ZIP is not set to true. Keeping the ZIP file."
fi

echo "All tasks completed successfully."
