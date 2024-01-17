## Description

This repo connects various physical devices to construct environments that comply with the [dm_env](https://github.com/google-deepmind/dm_env) API.

#### Content:
[scene/](ur_env/scene): physical environment description.
- [nodes/](ur_env/scene/nodes): various devices that can provide information or be actuated. 
  - [cameras/](ur_env/scene/nodes/cameras): vision perception.
  - [robot/](ur_env/scene/nodes/robot): muscles.
  - [sensors/](ur_env/scene/nodes/sensors): tactile perception.
- [scene.py](ur_env/scene/scene.py): container serves to aggregate and dispatch information from/to nodes.

[environment.py](ur_env/environment.py): task definition and RL environment wrapper.

[teleop/](ur_env/teleop): teleoperation via different controllers. This may require additional installation steps.

[remote.py](ur_env/remote.py): client and server connection for remote control. Can be used standalone with any dm_env.Environment.


## Installation
A host requires all the packages:
```
pip install "ur_env[all] @ git+https://github.com/RQC-Robotics/ur5-env.git"
```

**OR** you probably only need `RemoteEnvClient` to connect to the host from remote.
Then omit `[all]` or manually install `dm_env` and copy remote.py to a working dir.


## Examples

There are a few [examples](examples) to begin familiarizing with the rep.
