#!/usr/bin/env python3
"""MonoDepth2-style monocular depth estimation node.

IMPORTANT: Output is RELATIVE depth derived from disparity, not metric meters.
Do not use these values as calibrated Euclidean depth without external scale.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image

# Optional monodepth2 import — node degrades gracefully if unavailable
_HAS_MONODEPTH = False
ResnetEncoder = None  # type: ignore
DepthDecoder = None  # type: ignore


def _try_import_monodepth(extra_path: str = "") -> bool:
    global ResnetEncoder, DepthDecoder, _HAS_MONODEPTH
    if extra_path:
        sys.path.insert(0, os.path.expanduser(os.path.expandvars(extra_path)))
    try:
        from monodepth2.networks import ResnetEncoder as _RE, DepthDecoder as _DD  # type: ignore

        ResnetEncoder, DepthDecoder = _RE, _DD
        _HAS_MONODEPTH = True
        return True
    except Exception:
        try:
            from networks.resnet_encoder import ResnetEncoder as _RE  # type: ignore
            from networks.depth_decoder import DepthDecoder as _DD  # type: ignore

            ResnetEncoder, DepthDecoder = _RE, _DD
            _HAS_MONODEPTH = True
            return True
        except Exception:
            _HAS_MONODEPTH = False
            return False


class DepthEstimatorNode:
    def __init__(self) -> None:
        rospy.init_node("depth_estimator_node")
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() and not rospy.get_param("~force_cpu", False) else "cpu")
        self.input_w = int(rospy.get_param("~input_width", 640))
        self.input_h = int(rospy.get_param("~input_height", 192))
        self.headless = bool(rospy.get_param("~headless", True))
        self.log_every = int(rospy.get_param("~log_every_n_frames", 30))
        self.enabled = bool(rospy.get_param("~enabled", True))

        mono_path = rospy.get_param("~monodepth2_path", "")
        if mono_path:
            sys.path.insert(0, os.path.expanduser(os.path.expandvars(mono_path)))
        # Also allow sibling monodepth2 in catkin src
        catkin_src = os.path.expanduser("~/catkin_ws/src/monodepth2")
        if os.path.isdir(catkin_src):
            sys.path.insert(0, catkin_src)

        self.model_ready = False
        self.encoder = None
        self.decoder = None

        if self.enabled and _try_import_monodepth():
            self._load_weights()
        elif self.enabled:
            rospy.logwarn(
                "MonoDepth2 networks not importable. Depth node will publish empty markers. "
                "Install monodepth2 on PYTHONPATH or set ~monodepth2_path."
            )

        image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw/compressed")
        self.sub = rospy.Subscriber(
            image_topic, CompressedImage, self._cb, queue_size=1, buff_size=2**24
        )
        self.depth_pub = rospy.Publisher("~depth", Image, queue_size=1)
        self.depth_color_pub = rospy.Publisher("~depth_colormap", Image, queue_size=1)
        self._frames = 0
        rospy.loginfo(
            "depth_estimator_node ready device=%s model_ready=%s (output=RELATIVE depth)",
            self.device,
            self.model_ready,
        )

    def _load_weights(self) -> None:
        from vision_arm_control.model_config import load_model_paths

        model_root = rospy.get_param("~model_dir", "") or None
        paths = load_model_paths(
            model_root=model_root,
            encoder_name=rospy.get_param("~encoder_filename", "encoder.pth"),
            depth_name=rospy.get_param("~depth_filename", "depth.pth"),
        )
        missing = paths.missing()
        if missing:
            rospy.logwarn("Missing model files: %s — depth inference disabled", missing)
            return
        try:
            self.encoder = ResnetEncoder(18, False)
            self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales=range(4))
            enc_sd = torch.load(str(paths.encoder), map_location=self.device)
            # Filter unexpected keys (historical checkpoints)
            if isinstance(enc_sd, dict):
                filtered = {k: v for k, v in enc_sd.items() if k in self.encoder.state_dict()}
                self.encoder.load_state_dict(filtered, strict=False)
            dep_sd = torch.load(str(paths.depth), map_location=self.device)
            self.decoder.load_state_dict(dep_sd, strict=False)
            self.encoder.to(self.device).eval()
            self.decoder.to(self.device).eval()
            self.model_ready = True
            rospy.loginfo("Loaded encoder=%s depth=%s", paths.encoder, paths.depth)
        except Exception as exc:  # noqa: BLE001
            rospy.logerr("Failed to load depth models: %s", exc)
            self.model_ready = False

    def _cb(self, msg: CompressedImage) -> None:
        if not self.model_ready:
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return
            depth = self._infer(frame)
            if depth is None:
                return
            # Publish float relative depth as 32FC1
            depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding="32FC1")
            depth_msg.header = msg.header
            self.depth_pub.publish(depth_msg)

            # Colormap for visualization only
            depth_u8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
            color_msg = self.bridge.cv2_to_imgmsg(color, encoding="bgr8")
            color_msg.header = msg.header
            self.depth_color_pub.publish(color_msg)
            if not self.headless:
                cv2.imshow("depth", color)
                cv2.waitKey(1)
        except Exception as exc:  # noqa: BLE001
            rospy.logerr_throttle(5.0, "depth callback error: %s", exc)
        self._frames += 1
        if self._frames % max(self.log_every, 1) == 0:
            rospy.loginfo_throttle(10.0, "depth frames=%d", self._frames)

    def _infer(self, image_bgr: np.ndarray):
        assert self.encoder is not None and self.decoder is not None
        img = cv2.resize(image_bgr, (self.input_w, self.input_h))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.encoder(img)
            outputs = self.decoder(features)
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp,
                (image_bgr.shape[0], image_bgr.shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            # Relative depth proxy (1/disparity) — NOT metric meters
            depth = 1.0 / torch.clamp(disp_resized.squeeze(), min=1e-6)
            return depth.detach().cpu().numpy()


if __name__ == "__main__":
    try:
        DepthEstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
