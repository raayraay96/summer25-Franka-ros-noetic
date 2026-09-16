"""Verify the shell gate's success/failure contract without installing ROS."""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "topic_status,output,expected", [(0, "---\n---\n---\n", 0), (124, "", 1), (0, "", 1)]
)
def test_smoke_requires_controller_messages(tmp_path, topic_status, output, expected):
    setup = tmp_path / "setup.bash"
    setup.write_text("")
    commands = {
        "rospack": "exit 0",
        "rosparam": 'test -f "$FAKE_READY"',
        "roslaunch": 'touch "$FAKE_READY"; exec sleep 10',
        "rostopic": 'cat "$FAKE_TOPIC_OUTPUT"; exit "$FAKE_TOPIC_STATUS"',
    }
    for name, body in commands.items():
        path = tmp_path / name
        path.write_text("#!/usr/bin/env bash\n" + body + "\n")
        path.chmod(0o755)
    messages = tmp_path / "messages"
    messages.write_text(output)
    env = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        ROS_SETUP_FILE=str(setup),
        CATKIN_SETUP_FILE=str(setup),
        FAKE_READY=str(tmp_path / "ready"),
        FAKE_TOPIC_OUTPUT=str(messages),
        FAKE_TOPIC_STATUS=str(topic_status),
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/ci_smoke_noetic.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert (result.returncode == 0) == (expected == 0), result.stdout + result.stderr
    assert ("passed" in result.stdout) == (expected == 0)
