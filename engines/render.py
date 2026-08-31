import cv2
import numpy as np
import trimesh
import pyrender

MODEL_PATH = "assets/arrow.glb"

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
        follow_hand_rotation=True,
        collision_margin=0.05,
        collision_color=(0, 100, 255),
        collision_tint_strength=0.35,
        glow_strength=0.7,
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
        self.follow_hand_rotation = follow_hand_rotation

        self.collision_margin = collision_margin
        self.collision_color = np.array(
            collision_color,
            dtype=np.float32
        )

        self.collision_tint_strength = collision_tint_strength
        self.glow_strength = glow_strength

        self.is_colliding = False
        self.effect_phase = 0.0

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
    def euclid_distance(a, b):
        return np.sqrt(a**2 + b**2)
    
    def clamp_to_wrist(self, hand_landmarks, max_distance=0.12):
        palm_ids = [0, 5, 9, 13, 17]
        center_x = np.mean([hand_landmarks[i].x for i in palm_ids])
        center_y = np.mean([hand_landmarks[i].y for i in palm_ids])
        wrist = hand_landmarks[0]

        dx = center_x - wrist.x
        dy = center_y - wrist.y

        distance = self.euclid_distance(dx, dy)
        if distance <= max_distance:
            return center_x, center_y

        direction_x = dx / distance
        direction_y = dy / distance

        center_x = wrist.x + direction_x * max_distance
        center_y = wrist.y + direction_y * max_distance

        return center_x, center_y

    @staticmethod
    def _normalize(vector):
        norm = np.linalg.norm(vector)
        if norm < 1e-6:
            return vector

        return vector / norm

    def _hand_rotation(self, hand_landmarks):
        wrist = hand_landmarks[0]
        index = hand_landmarks[5]
        pinky = hand_landmarks[17]

        p0 = np.array([
            wrist.x,
            -wrist.y,
            -wrist.z
        ], dtype=np.float32)

        p5 = np.array([
            index.x,
            -index.y,
            -index.z
        ], dtype=np.float32)

        p17 = np.array([
            pinky.x,
            -pinky.y,
            -pinky.z
        ], dtype=np.float32)

        palm_center = (p5 + p17) / 2.0
        y_hint = self._normalize(palm_center - p0)

        x_axis = self._normalize(p5 - p17)
        z_axis = self._normalize(np.cross(x_axis, y_hint))
        y_axis = self._normalize(np.cross(z_axis, x_axis))

        rotation = np.column_stack([x_axis, y_axis, z_axis])

        return rotation.astype(np.float32)

    def _check_torso_collision(self, center_x, center_y, pose_landmarks):
        if pose_landmarks is None:
            return False

        if len(pose_landmarks) <= 24:
            return False

        ids = [11, 12, 23, 24]

        xs = [
            pose_landmarks[i].x
            for i in ids
        ]

        ys = [
            pose_landmarks[i].y
            for i in ids
        ]

        left = min(xs) - self.collision_margin
        right = max(xs) + self.collision_margin

        top = min(ys) - self.collision_margin
        bottom = max(ys) + self.collision_margin

        inside_x = left <= center_x <= right
        inside_y = top <= center_y <= bottom

        return inside_x and inside_y

    def compute_hand_transform(self, hand_landmarks, pose_landmarks=None):
        center_x, center_y = self.clamp_to_wrist(hand_landmarks)
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        pinch_distance = self.euclid_distance((thumb_tip.x - index_tip.x), (thumb_tip.y - index_tip.y))

        x_range = 2.2
        y_range = 1.6
        hand_z = hand_landmarks[9].z

        x = (center_x - 0.5) * x_range
        y = (0.5 - center_y) * y_range
        z = np.interp(
            hand_z,
            [-0.3, 0.2],
            [1.0, -1.0]
        )

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

        rotation = self._hand_rotation(hand_landmarks)
        self.is_colliding = self._check_torso_collision(center_x, center_y, pose_landmarks)

        return x, y, z, self.current_scale, rotation

    def build_pose(self, angle, x, y, z, scale, hand_rotation=None):
        rotation_y = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ], dtype=np.float32)

        if self.follow_hand_rotation and hand_rotation is not None:
            rotation = hand_rotation @ rotation_y
        else:
            rotation = rotation_y

        if self.is_colliding:
            self.effect_phase += 0.25

            pulse = 1.0 + 0.12 * (
                0.5 + 0.5 * np.sin(self.effect_phase)
            )

            scale *= pulse

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation * scale

        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z

        return pose

    def update_pose(self, hand_landmarks, angle, pose_landmarks=None):
        x, y, z, scale, hand_rotation = self.compute_hand_transform(hand_landmarks, pose_landmarks)
        pose = self.build_pose(angle, x, y, z, scale, hand_rotation)

        for node in self.nodes:
            self.scene.set_pose(node, pose)

        return self.is_colliding

    def render(self, width, height, flags=pyrender.RenderFlags.RGBA):
        self.resize(width, height)
        return self.renderer.render(self.scene, flags=flags)

    def render_overlay(self, frame: np.ndarray, flags=pyrender.RenderFlags.RGBA):
        frame_h, frame_w = frame.shape[:2]
        color, depth = self.render(frame_w, frame_h, flags=flags)

        bgr = color[:, :, :3].astype(np.float32)[:, :, ::-1]
        alpha = color[:, :, 3].astype(np.float32) / 255.0

        background = frame.astype(np.float32)

        if self.is_colliding:
            bgr = bgr * (
                1.0 - self.collision_tint_strength
            ) + self.collision_color * self.collision_tint_strength

            glow = cv2.GaussianBlur(alpha, (0, 0), sigmaX=12, sigmaY=12)
            glow = np.clip(glow - alpha, 0.0, 1.0)

            background = background + glow[:, :, None] * self.collision_tint_strength * self.collision_color

        output = bgr * alpha[:, :, None] + background * (1.0 - alpha[:, :, None])

        return np.clip(output, 0, 255).astype(np.uint8), depth

    def close(self):
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None

    def get_object_bbox(self, depth, padding=5):
        mask = depth > 0
        ys, xs = np.where(mask)

        if len(xs) == 0 or len(ys) == 0:
            return None

        x1 = max(int(xs.min()) - padding, 0)
        y1 = max(int(ys.min()) - padding, 0)
    
        x2 = min(int(xs.max()) + padding, depth.shape[1] - 1)
        y2 = min(int(ys.max()) + padding, depth.shape[0] - 1)
    
        return x1, y1, x2, y2

    def draw_object_bbox(self, frame, depth, color=(0, 255, 0), thickness=2, padding=5):
        bbox = self.get_object_bbox(depth, padding)

        if bbox is None:
            return frame

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        return frame