from time import time

class FaceExpression:
    def __init__(self, min_durations=1.0):
        self.min_durations = min_durations
        self.blendshapes = {}
        self.start_times = {}
        self.triggered = {}

    def get_blendshapes(self, face_blendshapes):
        self.blendshapes = {
            item.category_name: item.score
            for item in face_blendshapes
        }

    def get(self, name):
        return self.blendshapes.get(name, 0.0)

    def update(self, face_blendshapes, conditions: list):
        self.get_blendshapes(face_blendshapes)
        now = time()

        for method_name, kwargs in conditions:
            if method_name not in self.triggered:
                self.triggered[method_name] = False

            method = getattr(self, method_name)
            condition = method(**kwargs)

            if condition:
                duration = now - self.start_times[method_name]

                if duration >= self.min_durations and not self.triggered[method_name]:
                    self.triggered[method_name] = True
                    return True
            else:
                self.triggered[method_name] = False

            return False

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
        return self._validate("is_brow_up", condition)
        