#!/bin/bash
# setup_mlflow_tunnel.sh: This script installs autossh (if not installed) and automatically sets up
# a systemd unit file for an SSH tunnel that forwards remote port 5000 (MLflow UI) to localhost:5000.
#
# The script then reloads systemd, enables, and starts the service.
# Make sure to run this script with appropriate privileges (e.g., using sudo).

set -e

# Check if autossh is installed; if not, install it.
if ! command -v autossh >/dev/null 2>&1; then
    echo "autossh is not installed. Installing autossh..."
    sudo apt-get update && sudo apt-get install -y autossh
else
    echo "autossh is already installed."
fi

# Get the current user name (this will be used in the unit file)
CURRENT_USER=$(whoami)
echo "Current user: $CURRENT_USER"

# Set the path to the unit file that will be created in /etc/systemd/system
UNIT_FILE="/etc/systemd/system/mlflow-tunnel.service"

# Build the content of the unit file (modify parameters if needed)
read -r -d '' UNIT_CONTENT <<EOF
[Unit]
Description=AutoSSH Tunnel for MLflow UI
After=network.target

[Service]
User=$CURRENT_USER
ExecStart=/usr/bin/autossh -M 0 -N -L 5000:localhost:5000 ${CURRENT_USER}@localhost
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Write the content to the unit file (using sudo)
echo "Creating systemd unit file at $UNIT_FILE..."
echo "$UNIT_CONTENT" | sudo tee $UNIT_FILE > /dev/null

# Reload the systemd daemon to apply changes
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable the newly created service for autostart
echo "Enabling mlflow-tunnel service..."
sudo systemctl enable mlflow-tunnel.service

# Start the service
echo "Starting mlflow-tunnel service..."
sudo systemctl start mlflow-tunnel.service

# Display the service status
echo "Service status:"
sudo systemctl status mlflow-tunnel.service --no-pager

echo "Tunnel is established. You can now connect to the MLflow UI via http://<your_server_ip>:5000"