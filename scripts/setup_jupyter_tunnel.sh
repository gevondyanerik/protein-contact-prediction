#!/usr/bin/env bash

################################################################################
# This script sets up an SSH tunnel for accessing JupyterLab remotely using
# AutoSSH. It ensures that AutoSSH is installed, creates a systemd service to
# maintain the tunnel, and starts the service to keep the connection alive.
#
# Steps:
# 1. Check for AutoSSH installation:
#    - If not installed, installs it via `apt-get`.
#
# 2. Create a systemd service:
#    - Defines a persistent SSH tunnel that forwards local port 8888 to a remote port.
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
# - Systemd service file at `/etc/systemd/system/jupyter-tunnel.service`.
# - SSH tunnel accessible via `http://<your_server_ip>:8888`.
#
# Notes:
# - This script requires sudo privileges to modify systemd services.
# - Ensure that the MLflow UI is running locally before establishing the tunnel.
################################################################################

# Exit immediately if any command exits with a non-zero status.
set -e

# Check if autossh is installed.
if ! command -v autossh >/dev/null 2>&1; then
    echo "autossh is not installed. Installing autossh..."
    # Update package lists and install autossh.
    sudo apt-get update && sudo apt-get install -y autossh
else
    echo "autossh is already installed."
fi

# Get the current username.
CURRENT_USER=$(whoami)
echo "Current user: $CURRENT_USER"

# Define the path for the systemd unit file that will manage the autossh tunnel.
UNIT_FILE="/etc/systemd/system/jupyter-tunnel.service"

# Create the content for the systemd unit file using a here-document.
# This unit file defines a service that establishes an autossh tunnel for JupyterLab.
read -r -d '' UNIT_CONTENT <<EOF
[Unit]
Description=AutoSSH Tunnel for JupyterLab
After=network.target

[Service]
User=$CURRENT_USER
# ExecStart runs autossh to create a tunnel from local port 8888 to remote port 8888.
# The -M 0 option disables monitoring; -N indicates no remote command; -L sets up the local port forwarding.
ExecStart=/usr/bin/autossh -M 0 -N -L 8888:localhost:8888 ${CURRENT_USER}@localhost
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Inform the user that the unit file is being created.
echo "Creating systemd unit file at $UNIT_FILE..."
# Write the unit content to the file using sudo tee (suppressing output).
echo "$UNIT_CONTENT" | sudo tee $UNIT_FILE > /dev/null

# Reload the systemd daemon to recognize the new unit file.
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable the new service so that it starts automatically at boot.
echo "Enabling jupyter-tunnel service..."
sudo systemctl enable jupyter-tunnel.service

# Start the service immediately.
echo "Starting jupyter-tunnel service..."
sudo systemctl start jupyter-tunnel.service

# Display the status of the service to confirm it is running.
echo "Service status:"
sudo systemctl status jupyter-tunnel.service --no-pager

# Inform the user that the tunnel is established.
echo "Tunnel is established: You can now connect to JupyterLab via http://<your_server_ip>:8888"
