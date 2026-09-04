#!/usr/bin/env bash
set -euo pipefail

# Install Flint systemd user timer for weekday startup before US market open
#
# Note: `loginctl enable-linger alex` must be run once so user timers
# continue running without an active login session. This is required
# because the Spark is a headless box that may reboot or log out.

USER="${USER:-alex}"

# Check if linger is enabled for this user
LINGER_STATUS=$(loginctl show-user "$USER" -p Linger 2>/dev/null || echo "Linger=no")
if [[ "$LINGER_STATUS" != "Linger=yes" ]]; then
    echo "Error: Linger is not enabled for user $USER"
    echo "Run this command once to enable user timers:"
    echo ""
    echo "    loginctl enable-linger $USER"
    echo ""
    exit 1
fi

# Create the systemd user directory if it doesn't exist
mkdir -p "$HOME/.config/systemd/user"

# Copy the unit files
cp "$(dirname "$0")/flint.service" "$HOME/.config/systemd/user/"
cp "$(dirname "$0")/flint.timer" "$HOME/.config/systemd/user/"

# Reload systemd user daemon
systemctl --user daemon-reload

# Enable and start the timer
systemctl --user enable --now flint.timer

# Show the timer status
echo "Flint timer installed and running:"
systemctl --user list-timers flint.timer
