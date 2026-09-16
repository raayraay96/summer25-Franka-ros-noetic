"""Exercise the real ROS callback with lightweight message/transport stand-ins."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/vision_arm_control/src"))
from vision_arm_control.safety import SafetyConfig, SafetyMonitor  # noqa: E402
from vision_arm_control.workspace_limits import AxisAlignedBounds  # noqa: E402


class Stamp:
    def __init__(self, value):
        self.value = value

    def to_sec(self):
        return self.value


class Pose:
    def __init__(self, stamp=0.0):
        self.header = SimpleNamespace(stamp=Stamp(stamp), frame_id="panda_link0")
        self.pose = SimpleNamespace(
            position=SimpleNamespace(x=0.45, y=0.0, z=0.45), orientation=SimpleNamespace()
        )


@pytest.fixture
def node(monkeypatch):
    modules = {
        "rospy": {"Time": SimpleNamespace(now=lambda: Stamp(10.0))},
        "geometry_msgs": {},
        "geometry_msgs.msg": {"PoseStamped": Pose},
        "std_msgs": {},
        "std_msgs.msg": {"Bool": SimpleNamespace, "String": SimpleNamespace},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "safety_node_under_test", ROOT / "src/vision_arm_control/scripts/safety_monitor_node.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = module.SafetyMonitorNode.__new__(module.SafetyMonitorNode)
    instance.monitor = SafetyMonitor(SafetyConfig(AxisAlignedBounds(0.25, 0.75, -0.4, 0.4, 0.05, 0.8)))
    instance.commands = []
    instance.statuses = []
    instance.pub = SimpleNamespace(publish=instance.commands.append)
    instance.status_pub = SimpleNamespace(publish=instance.statuses.append)
    instance._publish_telemetry = lambda payload: None
    return instance


@pytest.mark.parametrize("stamp", [9.0, 11.0, float("nan"), float("inf"), 0.0])
def test_invalid_input_stamp_never_reaches_controller(node, stamp):
    node._cb(Pose(stamp))
    assert node.commands == []
    assert node.statuses[-1].data == "pose_timeout"


def test_valid_pose_preserves_acquisition_timestamp(node):
    message = Pose(9.9)
    node._cb(message)
    assert len(node.commands) == 1
    assert node.commands[0].header.stamp.to_sec() == 9.9
    assert message.header.stamp.to_sec() == 9.9


@pytest.fixture
def mapper(node, monkeypatch):
    import numpy as np
    from vision_arm_control.coordinate_mapping import CameraIntrinsics
    from vision_arm_control.pose_timeout import PoseTimeoutMonitor
    from vision_arm_control.trajectory_filter import LowPassFilter3D, VelocityLimiter3D

    for name in ("tf", "tf.transformations"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["rospy"].Time.from_sec = Stamp
    spec = importlib.util.spec_from_file_location(
        "mapper_under_test", ROOT / "src/vision_arm_control/scripts/human_to_robot_mapper_node.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = module.HumanToRobotMapperNode.__new__(module.HumanToRobotMapperNode)
    instance.pose_timeout = PoseTimeoutMonitor(0.5)
    instance.min_conf = 0.5
    instance.intrinsics = CameraIntrinsics(600, 600, 320, 240, 640, 480)
    instance.T_base_camera = np.eye(4)
    instance.rel_scale, instance.rel_offset = 1.0, 0.8
    instance.use_metric_depth = False
    instance.workspace_origin = [0.45, 0, 0.45]
    instance.workspace_scale = [0.55, 0.55, 0.35]
    instance.mode = "shoulder_relative"
    instance.workspace_mode = "reject"
    instance.bounds = node.monitor.config.workspace
    instance.lpf = LowPassFilter3D(0.3)
    instance.vlim = VelocityLimiter3D(0.2)
    instance._last_t = None
    instance.base_frame = "panda_link0"
    instance.fixed_q = [1, 0, 0, 0]
    instance.commands, instance.rejections = [], []
    instance.pub = SimpleNamespace(publish=instance.commands.append)
    instance.reject_pub = SimpleNamespace(publish=instance.rejections.append)
    return instance


def landmark_message(stamp):
    import json

    return SimpleNamespace(
        data=json.dumps(
            {
                "header": {"stamp": stamp},
                "pose_detected": True,
                "landmarks": [
                    {"name": "right_wrist", "x": 0.5, "y": 0.4, "confidence": 1},
                    {"name": "right_shoulder", "x": 0.45, "y": 0.35, "confidence": 1},
                ],
            }
        )
    )


@pytest.mark.parametrize("stamp", [9.0, 11.0, None, "bad", float("nan")])
def test_mapper_rejects_stale_or_invalid_acquisition_time(mapper, stamp):
    mapper._cb(landmark_message(stamp))
    assert mapper.commands == []
    assert mapper.rejections[-1].data in {"invalid_pose_timestamp", "pose_timeout"}


def test_mapper_preserves_acquisition_time(mapper):
    mapper._cb(landmark_message(9.9))
    assert len(mapper.commands) == 1
    assert mapper.commands[0].header.stamp.to_sec() == 9.9
