import numpy as np
import rtde_control
import rtde_receive

from ur_env.teleop.vive import ViveController


center = np.float32([-0.5, 0, 0.3, 2.2, 2.2, 0])
xs_world = []
for _ in range(9):
    offset = [np.random.uniform(-0.15, 0.15, 3), np.random.uniform(-0.4, 0.4, 3)]
    xs_world.append(offset)
xs_world = center + np.asarray(xs_world)

HOST = "localhost"
rtde_r = rtde_receive.RTDEReceiveInterface(HOST)
rtde_c = rtde_control.RTDEControlInterface(HOST)
controller = ViveController("calibation.npz")
controller.calibrate(xs_world, rtde_c.moveL)


def actuate(vstate):
    if state.menu_button:
        return rtde_c.servoStop()
    vel = vstate.velocity
    speed = max(np.linalg.norm(vel), 1e-3)
    axang = vstate.rotation.as_rotvec()
    pos = np.asarray(rtde_r.getActualTCPPose())
    new_pos = pos[:3] + .01 * vel / speed
    new_pose = np.concatenate([new_pos, axang])
    # pose, speed, acceleration, time, lookahead_time, gain
    rtde_c.servoL(list(new_pose), 0., 0., 0.01, 0.2, 100.)


print("Ready")
while True:
    state = controller.read_state()
    actuate(state)
    if state.menu_button:
        break
