import numpy as np
import trimesh
import pyrender

MODEL_PATH = "assets/orangutan.glb"

class HandObjectRenderer:
    def __init__(
        self,
        model_path=MODEL_PATH,
        viewport_width=640,
        viewport_height=480,
        min_pinch_distance=0.02,
        max_pinch_distance=0.25,
        min_scale=0.3,
        max_scale=1.5,
        scale_smoothing=0.3,
    ):
        self.model_path = model_path
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.min_pinch_distance = min_pinch_distance
        self.max_pinch_distance = max_pinch_distance
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_smoothing = scale_smoothing
        self.current_scale = min_scale

        self.scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],
            ambient_light=[0.4, 0.4, 0.4],
        )
        self.nodes = []

        self._load_model()
        self._add_camera()
        self._add_light()
        self.renderer = self._create_renderer(viewport_width, viewport_height)

    def _load_model(self):
        tm_scene = trimesh.load(self.model_path)

        if isinstance(tm_scene, trimesh.Scene):
            for geometry in tm_scene.geometry.values():
                mesh = pyrender.Mesh.from_trimesh(geometry, smooth=False)
                self.nodes.append(self.scene.add(mesh))
            return

        mesh = pyrender.Mesh.from_trimesh(tm_scene, smooth=False)
        self.nodes.append(self.scene.add(mesh))

    def _add_camera(self):
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[2, 3] = 3.0
        self.scene.add(camera, pose=camera_pose)

    def _add_light(self):
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        light_pose = np.eye(4, dtype=np.float32)
        light_pose[2, 3] = 1.0
        self.scene.add(light, pose=light_pose)

    def _create_renderer(self, width, height):
        return pyrender.OffscreenRenderer(
            viewport_width=width,
            viewport_height=height,
        )

    def resize(self, width, height):
        if width == self.viewport_width and height == self.viewport_height:
            return

        self.renderer.delete()
        self.renderer = self._create_renderer(width, height)
        self.viewport_width = width
        self.viewport_height = height

    @staticmethod
    def _palm_center(hand_landmarks):
        palm_ids = [0, 5, 9, 13, 17]
        center_x = np.mean([hand_landmarks[i].x for i in palm_ids])
        center_y = np.mean([hand_landmarks[i].y for i in palm_ids])
        return center_x, center_y

    @staticmethod
    def _pinch_distance(hand_landmarks):
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        return ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

    def compute_hand_transform(self, hand_landmarks):
        center_x, center_y = self._palm_center(hand_landmarks)
        pinch_distance = self._pinch_distance(hand_landmarks)

        x_range = 2.2
        y_range = 1.6

        x = (center_x - 0.5) * x_range
        y = (0.5 - center_y) * y_range

        target_scale = np.interp(
            pinch_distance,
            [self.min_pinch_distance, self.max_pinch_distance],
            [self.min_scale, self.max_scale],
        )
        target_scale = float(np.clip(target_scale, self.min_scale, self.max_scale))

        self.current_scale = (
            self.scale_smoothing * target_scale
            + (1.0 - self.scale_smoothing) * self.current_scale
        )

        return x, y, self.current_scale

    @staticmethod
    def build_pose(angle, x, y, scale):
        pose = np.array([
            [np.cos(angle), 0, np.sin(angle), x],
            [0, 1, 0, y],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        pose[:3, :3] *= scale
        return pose

    def update_pose(self, hand_landmarks, angle):
        x, y, scale = self.compute_hand_transform(hand_landmarks)
        pose = self.build_pose(angle, x, y, scale)

        for node in self.nodes:
            self.scene.set_pose(node, pose)

    def render(self, width, height, flags=pyrender.RenderFlags.RGBA):
        self.resize(width, height)
        return self.renderer.render(self.scene, flags=flags)

    def render_overlay(self, frame, flags=pyrender.RenderFlags.RGBA):
        frame_h, frame_w = frame.shape[:2]
        color, depth = self.render(frame_w, frame_h, flags=flags)

        rgb = color[:, :, :3].astype(np.float32)
        alpha = (color[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
        output = rgb * alpha + frame.astype(np.float32) * (1.0 - alpha)

        return np.clip(output, 0, 255).astype(np.uint8), depth

    def close(self):
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None
