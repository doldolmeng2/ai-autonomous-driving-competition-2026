#!/usr/bin/env bash

set -eo pipefail

WORKSPACE="$HOME/osy/260801/ai-autonomous-driving-competition-2026"

if [[ ! -d "$WORKSPACE" ]]; then
    echo "Workspace not found: $WORKSPACE" >&2
    exit 1
fi

source /opt/ros/humble/setup.bash
cd "$WORKSPACE"

if [[ ! -f install/setup.bash ]]; then
    echo "Workspace is not built. Run colcon build --symlink-install first." >&2
    exit 1
fi

source install/setup.bash

sensor_launch_pid=''

cleanup_sensor_launch() {
    if [[ -n "$sensor_launch_pid" ]] && kill -0 "$sensor_launch_pid" 2>/dev/null; then
        kill -INT "$sensor_launch_pid" 2>/dev/null || true
        wait "$sensor_launch_pid" 2>/dev/null || true
    fi
}

start_sensors_and_wait() {
    local required_topics="${1:-}"

    trap cleanup_sensor_launch EXIT
    trap 'exit 130' INT TERM

    ros2 launch sensor_topic sensors.launch.py &
    sensor_launch_pid=$!

    if [[ -n "$required_topics" ]]; then
        ros2 run sensor_topic sensor_readiness_node --ros-args \
            -p "required_topics:=$required_topics"
    else
        ros2 run sensor_topic sensor_readiness_node
    fi

    if ! kill -0 "$sensor_launch_pid" 2>/dev/null; then
        echo 'Sensor launch exited before required sensors became ready.' >&2
        exit 1
    fi
}

case "${1:-}" in
    lane_time)
        exec ros2 launch lane_main timed_lane_main.launch.py
        ;;
    lane_mission)
        exec ros2 launch lane_main mission_lane_main.launch.py
        ;;
    parking)
        exec ros2 launch parking parking.launch.py
        ;;
    manual_bag)
        start_sensors_and_wait
        ros2 launch sensor_utils sensors_controller_bag.launch.py
        ;;
    tuner_ycrcb)
        start_sensors_and_wait "['/camera/high/image_raw']"
        ros2 run sensor_utils ycrcb_tuner_node
        ;;
    tuner_hsv)
        start_sensors_and_wait "['/camera/high/image_raw']"
        ros2 run sensor_utils hsv_tuner_node
        ;;
    tuner_hsv_ycrcb)
        start_sensors_and_wait "['/camera/high/image_raw']"
        ros2 run sensor_utils hsv_ycrcb_tuner_node
        ;;
    *)
        echo "Usage: $0 {lane_time|lane_mission|parking|manual_bag|tuner_ycrcb|tuner_hsv|tuner_hsv_ycrcb}" >&2
        exit 2
        ;;
esac
