# MediaPipe Human Interaction Engine

A MediaPipe playground for experimenting with **camera-based human-computer interaction** using hand gestures, facial expressions, segmentation, and real-time visual effects.

The project currently uses **MediaPipe, OpenCV, and NumPy** to control camera effects through human interaction.

## Current Interaction

Two interactions are currently used:

* **Pinch** → randomly changes the active filter.
* **Eyebrow raise** → switches to the next interaction mode.

The basic flow is:

```text
Camera
  ↓
MediaPipe
  ├── Hand Landmarks → Pinch Detection
  ├── Face Blendshapes → Eyebrow Detection
  └── Segmentation
  ↓
Interaction
  ├── Pinch → Random Filter
  └── Eyebrow Raise → Change Mode
  ↓
Visual Effect
```

## Interaction Modes

The project currently provides several ways to apply visual effects.

| Mode                    | Description                                                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **RectanglePipeline**   | Creates a rectangular interaction area using hand positions. The selected filter is applied inside the rectangle.                      |
| **PolyPipeline**        | Creates a polygon from multiple hand landmarks, allowing the filtered area to follow a more dynamic hand-defined shape.                |
| **CirclePipeline**      | Uses a circular interaction area instead of a rectangle or polygon.                                                                    |
| **SpotlightPipeline**   | Creates a spotlight-like region that follows the interaction area while visually separating it from the rest of the frame.             |
| **SelfieSegmentation**  | Uses MediaPipe segmentation to separate the person from the background, allowing effects to be applied specifically to the foreground. |
| **SpotlightBackground** | Extends the spotlight concept by applying an effect to the background while maintaining a separate focused area.                       |

Each mode uses the same camera input and MediaPipe detection, but interprets the detected landmarks or segmentation mask differently.

## Installation

Create and activate a Conda environment:

```bash
conda create -n <env-name> python=3.12
conda activate <env-name>
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Make sure your webcam is available and not currently being used by another application.

## Tech Stack

* Python
* MediaPipe
* OpenCV
* NumPy

---