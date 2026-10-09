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

# flint-start sets FLINT_COMPILE=1, and on the first train step Triton builds a CUDA helper
# against Python.h. Without the venv Python's dev headers that step fails and Flint trains
# eager (about half speed on the GB10). Nothing else notices until then, so say it here;
# an OS reinstall drops the package.
PY="$(dirname "$0")/../../.venv/bin/python"
if [[ -x "$PY" ]]; then
    read -r INC PKG < <("$PY" -c 'import sys, sysconfig; print(sysconfig.get_paths()["include"], f"python{sys.version_info[0]}.{sys.version_info[1]}-dev")')
    if [[ ! -f "$INC/Python.h" ]]; then
        echo "Warning: $INC/Python.h is missing, so torch.compile cannot build Triton's CUDA helper"
        echo "and Flint will train eager. Install the headers for the venv's Python:"
        echo ""
        echo "    sudo apt install $PKG"
        echo ""
    fi
fi

# Create the systemd user directory if it doesn't exist
mkdir -p "$HOME/.config/systemd/user"

# Install the launcher the service runs, then the unit files
mkdir -p "$HOME/.local/bin"
install -m 0755 "$(dirname "$0")/flint-start" "$HOME/.local/bin/flint-start"
cp "$(dirname "$0")/flint.service" "$HOME/.config/systemd/user/"
cp "$(dirname "$0")/flint.timer" "$HOME/.config/systemd/user/"

# Reload systemd user daemon
systemctl --user daemon-reload

# Enable and start the timer
systemctl --user enable --now flint.timer

# Show the timer status
echo "Flint timer installed and running:"
systemctl --user list-timers flint.timer
