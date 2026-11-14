# observations

What the robot measures, with who may read it. The bottom layer: it imports nothing else from PhyGS.

| File | Holds |
| --- | --- |
| `types.py` | The measurement types (`ObservationPacket`, `CameraObservation`, `JointObservation`, `ImuObservation`, `LocalizationObservation`), `Stamp` and `Transform` (every transform maps source coordinates into target), and the permissions (`ObservationConfig`, `CameraPermission`, `ObservationAccess`). |
| `localization.py` | The localization provider boundary; ground truth is an explicit, privileged mode. |
| `front_view.py` | The Front view: a robot's two front cameras joined into one forward-looking image. |
| `color_detection.py` | Reference colour detectors for diagnostic targets, not semantic detection. |

Entry points: `from phygs.observations import ...` is not an agent surface; agents get packets
from `session.observe()` and `session.stream()` ([docs/observations.md](../../docs/observations.md);
the views and permissions that filter them: [docs/pipelines.md](../../docs/pipelines.md)).

Up: [phygs/README.md](../README.md). Tests: `tests/test_observations.py`, `tests/test_pipeline.py`,
`tests/test_color_detection.py`, `tests/test_ui_stitch.py` (the front view).
