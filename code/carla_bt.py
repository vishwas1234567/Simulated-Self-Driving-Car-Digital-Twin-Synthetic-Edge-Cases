"""Run the project's behaviour tree in CARLA (>= 0.9.14, synchronous mode, ground-truth pedestrian state).
Needs a running CARLA server and `pip install carla`. NOT executed in the project sandbox (no CARLA available).
CARLA frame: x forward, y right  ->  project frame: y_sim = -(y - y0).
Usage: python carla_bt.py --host localhost --port 2000"""
import argparse
import numpy as np
from sim import DT, EGO_L
from stack import build_tree, derive, tick, Ctx

PED_X, PED_Y0, PED_SPEED = 60.0, 5.5, 1.6


class View:  # minimal stand-in for BatchSim fields used by derive()
    pc_len, pc_x0, pc_x1 = np.zeros(1), np.zeros(1), np.zeros(1)

    def __init__(s, x, v): s.x, s.v = np.array([x]), np.array([v])


def main(a):
    import carla
    cl = carla.Client(a.host, a.port); cl.set_timeout(20.0); w = cl.get_world()
    st = w.get_settings(); st.synchronous_mode, st.fixed_delta_seconds = True, DT; w.apply_settings(st)
    lib, o = w.get_blueprint_library(), w.get_map().get_spawn_points()[0].location
    ego = w.spawn_actor(lib.find("vehicle.tesla.model3"), carla.Transform(carla.Location(o.x, o.y, o.z + 0.5)))
    ped = w.spawn_actor(lib.filter("walker.pedestrian.*")[0], carla.Transform(carla.Location(o.x + PED_X, o.y - PED_Y0, o.z + 1.0)))
    hit = []; col = w.spawn_actor(lib.find("sensor.other.collision"), carla.Transform(), attach_to=ego); col.listen(lambda e: hit.append(e))
    tree, started = build_tree(), False
    try:
        for k in range(int(30 / DT)):
            v3, pl = ego.get_velocity(), ego.get_location(); pp = ped.get_location()
            x, v = pl.x - o.x, float(np.hypot(v3.x, v3.y))
            if x > PED_X - 25 and not started:
                ped.apply_control(carla.WalkerControl(carla.Vector3D(0, 1, 0), PED_SPEED)); started = True   # +y_carla = -y_sim
            pv = ped.get_velocity()
            est = (np.ones(1, bool), np.array([-(pp.y - o.y)]), np.array([-pv.y]), np.array([pp.x - o.x]), np.array([0.4]))
            c = Ctx(); derive(c, View(x, v), est, np.zeros(1, bool), np.full(1, 8.0))
            acc = float(tick(tree, c, 1)[0])
            ego.apply_control(carla.VehicleControl(throttle=float(np.clip(acc / 3.0, 0, 1)), brake=float(np.clip(-acc / 8.0, 0, 1))))
            w.tick()
            if hit or x > PED_X + 8: break
        print("collision" if hit else "no collision", "| final x = %.1f m" % x)
    finally:
        col.destroy(); ped.destroy(); ego.destroy(); st.synchronous_mode = False; w.apply_settings(st)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=2000)
    main(ap.parse_args())
