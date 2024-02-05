import numpy as np
import replica_gt_renderer.transformation as transformation
import yaml
import yaml
import matplotlib.pyplot as plt
import cv2
from replica_gt_renderer.habitat_renderer import set_agent_position, init_habitat
from replica_gt_renderer.habitat_renderer import render as render_habitat
import numpy as np
from scipy.spatial.transform import Rotation as R

def replica_load_poses(path):
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for i in range(2000):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        poses.append(c2w)
    return np.array(poses)

def main():      
    # Initialize habitat-sim
    config_file = "configs/habitat_config.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    sim, hs_cfg, config = init_habitat(config)
    sim.initialize_agent(config["default_agent"])
    poses = replica_load_poses("traj.txt")
    c2w = poses[10]
    print(sim)
    r = R.from_euler('z', 1, degrees=True)
    T = np.eye(4)
    T[:3,:3] = r.as_matrix()
    
    for i in range(2000):
        c2w = T@c2w
        set_agent_position(sim, transformation.Twc_to_Thc(c2w))
        # set_agent_position(sim, c2w)
        observation = render_habitat(sim, config)
        gt_rgb = observation["color_sensor"]
        gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_BGR2RGB)
        gt_depth = observation["depth_sensor"]
        
        cv2.imshow("rgb", gt_rgb)
        cv2.imshow("depth", gt_depth/10)
        cv2.waitKey(3)

if __name__ =="__main__":
    main()