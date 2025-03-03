#!/usr/bin/env bash

################################################################################
# This script sets up an SSH tunnel for accessing the MLflow UI remotely using
# AutoSSH. It ensures that AutoSSH is installed, creates a systemd service to
# maintain the tunnel, and starts the service to keep the connection alive.
#
# Steps:
# 1. Check for AutoSSH installation:
#    - If not installed, installs it via `apt-get`.
#
# 2. Create a systemd service:
#    - Defines a persistent SSH tunnel that forwards local port 5000 to a remote port.
#    - Runs the service as the current user.
#
# 3. Configure and start the service:
#    - Saves the systemd unit file at `/etc/systemd/system/mlflow-tunnel.service`.
#    - Reloads systemd to register the new service.
#    - Enables the service to start automatically on boot.
#    - Starts the service immediately.
#
# 4. Confirm the tunnel is active:
#    - Prints service status.
#    - Outputs connection instructions.
#
# Outputs:
# - Systemd service file at `/etc/systemd/system/mlflow-tunnel.service`.
# - SSH tunnel accessible via `http://<your_server_ip>:5000`.
#
# Notes:
# - This script requires sudo privileges to modify systemd services.
# - Ensure that the MLflow UI is running locally before establishing the tunnel.
################################################################################

# Exit immediately if any command returns a non-zero exit status.
set -e

# Check if autossh is installed. If not, install it.
if ! command -v autossh >/dev/null 2>&1; then
    echo "autossh is not installed. Installing autossh..."
    # Update package lists and install autossh.
    sudo apt-get update && sudo apt-get install -y autossh
else
    echo "autossh is already installed."
fi

# Get the current user name and display it.
CURRENT_USER=$(whoami)
echo "Current user: $CURRENT_USER"

# Define the path to the systemd unit file for the MLflow tunnel service.
UNIT_FILE="/etc/systemd/system/mlflow-tunnel.service"

# Create the content of the systemd unit file using a here-document.
read -r -d '' UNIT_CONTENT <<EOF
[Unit]
Description=AutoSSH Tunnel for MLflow UI
After=network.target

[Service]
User=$CURRENT_USER
# Start autossh to forward local port 5000 (MLflow UI) to remote port 5000.
ExecStart=/usr/bin/autossh -M 0 -N -L 5000:localhost:5000 ${CURRENT_USER}@localhost
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Inform the user that the unit file is being created.
echo "Creating systemd unit file at $UNIT_FILE..."
# Write the unit file content to the defined file (using sudo tee).
echo "$UNIT_CONTENT" | sudo tee $UNIT_FILE > /dev/null

# Reload systemd so that it recognizes the new unit file.
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable the service so that it starts on boot.
echo "Enabling mlflow-tunnel service..."
sudo systemctl enable mlflow-tunnel.service

# Start the mlflow-tunnel service immediately.
echo "Starting mlflow-tunnel service..."
sudo systemctl start mlflow-tunnel.service

# Display the current status of the service (without pager for immediate output).
echo "Service status:"
sudo systemctl status mlflow-tunnel.service --no-pager

# Final message to indicate the tunnel is established and provide connection instructions.
echo "Tunnel is established. You can now connect to the MLflow UI via http://<your_server_ip>:5000"
