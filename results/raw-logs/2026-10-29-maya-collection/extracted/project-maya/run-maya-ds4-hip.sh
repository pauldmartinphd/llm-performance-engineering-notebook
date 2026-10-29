#!/bin/sh
# starts the Project Maya dashboard and API (written by maya.py; ./maya.sh does the same)
cd /opt/project-maya || exit 1
exec /opt/project-maya/.venv/bin/python /opt/project-maya/maya.py --config /opt/project-maya/maya-ds4-hip.json --port 8080 "$@"
