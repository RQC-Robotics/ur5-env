# ur_env

## Description

This repo allows to connect various physical devices to construct RL environments that use dm-env API.

[rqcsuite/](ur_env/rqcsuite): collection of tasks for solving on a robot.

[scene/](ur_env/scene): process all the physical devices.

- [nodes/](ur_env/scene/nodes): various devices that can provide information or be controlled by an agent. 
  - [cameras/](ur_env/scene/nodes/cameras): vision perception.
  - [robot/](ur_env/scene/nodes/robot): muscles.
  - [sensors/](ur_env/scene/nodes/sensors): tactile perception.
- [scene.py](ur_env/scene/scene.py): hold, actuate and poll for observations every connected node.


[remote.py](ur_env/remote.py): client and server to establish connection between robot and remote host.
Can be used standalone with any dm_env.Environment.

[environment.py](ur_env/environment.py): RL task and environment. 


## Installation

Clone this repo and install all the listed packages and their requirements.
```
pip install .[all]
```

__OR__ you probably only need `RemoteEnvClient` to connect to a robot from remote.
Then install with `pip install .` or install `dm_env` and copy remote.py to a working dir.


## Examples

There are a few [examples](examples) to begin familiarizing with the rep.
