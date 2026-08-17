from time import time

class FaceExpression:
    def __init__(self, min_durations=1.0):
        self.min_durations = min_durations
        self.blendshapes = {}
        self.start_times = {}

    def get_blendshapes(self, face_blendshapes):
        self.blendshapes = {
            item.category_name: item.score
            for item in face_blendshapes
        }

    def get(self, name):
        return self.blendshapes.get(name, 0.0)

    def _validate(self, name, conditions):
        now = time()

        if not conditions:
            self.start_times.pop(name, None)
            return False

        if name not in self.start_times:
            self.start_times[name] = now
            return self.min_durations <= 0

        elapsed = now - self.start_times[name]
        return elapsed >= self.min_durations

    def get_durations(self, name):
        if name not in self.start_times:
            return 0.0

        return time() - self.start_times[name]

    def is_brow_up(self, threshold=0.01):
        condition = self.get("browDownLeft") <= threshold
        return self._validate("brow up", condition)
        