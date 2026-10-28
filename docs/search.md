# Object search on the shared runtime

`ObjectSearchTask` changes the goal from a destination to a target description.
The reference search explores with `robot.navigation.explore()` and walks to coverage viewpoints
with `robot.navigation.navigate_to(...)`, while a colour detector watches the camera. Navigation's
camera costmap and ground-truth localization are private service inputs. An agent can instead use
permitted `robot.locomotion.command_velocity(...)` calls and own its search logic. Search does not
require manipulation.

A target declaration is a camera name, capture stamp, and image pixel. The
agent-facing task endpoint is `tests/robot_tests/tasks/object_search_api.py`:

```python
camera = robot.observe().cameras["inspection"]
status = search_task.declare_target(camera.stamp, pixel_x, pixel_y)
```

The evaluator stores trusted camera depth, calibration, and the target's physical bounds at
each capture time. It unprojects the submitted pixel and nearby support pixels
using measured depth and calibration, then checks whether those rays terminate
on the target. An occluder in front of the target yields a false positive, and
an uncaptured, expired, or cross-episode stamp is rejected. The agent never
receives the target bounds through this task endpoint.

The house search check uses a magenta cube and a colour detector
(`helpers/color_detection.py`). That detector validates the lifecycle; it is not a general
object-recognition module and does not use the target pose. The evaluator does not trust the
detector's classification or a navigation success flag. Localization is ground truth; this is
recorded in the report.

From `/workspace/isaaclab`:

```bash
source /opt/explore_ws/install/setup.bash
# explore() over the unexplored kitchen, then a coverage pass, until the detector sees the cube.
./isaaclab.sh -p scripts/PhyGS/tests/robot_tests/components/check_nav2_pipeline.py \
  --output scripts/PhyGS/output_check/$(date +%Y%m%d-%H%M)_house_search

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./isaaclab.sh -p -m pytest \
  scripts/PhyGS/tests/test_object_search.py -q
```

The supported scope is one static target, one RGB-D camera and single-floor navigation. Future
tasks can change the target placement, viewpoint policy, detector, mapping/localization service
and action permissions without changing the evaluator's evidence contract.
