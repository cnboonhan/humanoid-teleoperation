# python3 ik.py --path Wiki-GRx-Models/GRX/GR1/GR1T2/urdf/GR1T2_fourier_hand_6dof.urdf  --port 8080 --flask-port 5000
# curl http://localhost:5000/config

import time
import tyro
import viser
import numpy as np

import pyroki as pk
from viser.extras import ViserUrdf
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
from yourdfpy import URDF
from _solve_ik_with_multiple_targets import solve_ik_with_multiple_targets
from flask import Flask, jsonify
import threading


# Global variables to store robot state
current_robot_config = None
actuated_joint_names = None

# Create Flask app
app = Flask(__name__)

@app.route('/config', methods=['GET'])
def get_robot_config():
    """Return the current robot configuration as JSON"""
    global current_robot_config, actuated_joint_names
    
    if current_robot_config is None or actuated_joint_names is None:
        return jsonify({"error": "Robot not initialized"}), 500
    
    # Create a dictionary mapping joint names to their current values
    config_dict = {
        "joint_config": dict(zip(actuated_joint_names, current_robot_config.tolist())),
        "timestamp": time.time()
    }
    
    return jsonify(config_dict)

def run_flask_server(flask_port=5000):
    """Run Flask server in a separate thread"""
    app.run(host='0.0.0.0', port=flask_port, debug=False, use_reloader=False)

def main(path: str, port: int, flask_port: int = 5000) -> None:
    global current_robot_config, actuated_joint_names
    
    urdf = URDF.load(path, load_collision_meshes=True, build_collision_scene_graph=True)
    
    robot = pk.Robot.from_urdf(urdf)
    server = viser.ViserServer(port=port)
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    
    ### Printing Robot Information
    print(f"\nAll links ({len(urdf.link_map)}):")
    for i, (link_name, link) in enumerate(urdf.link_map.items()):
        print(f"  {i:2d}: {link_name}")
     
    actuated_joints = [name for name, joint in urdf.joint_map.items() if joint.type != 'fixed']
    actuated_joint_names = actuated_joints  # Store globally for Flask API
    print(f"\nActuated joints ({len(actuated_joints)}): {actuated_joints}")
    
    ### Setting Initial Configuration for Robot Pose
    initial_config = []
    for joint_name, (lower, upper) in urdf_vis.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        initial_config.append(initial_pos)
    
    initial_config_array = np.array(initial_config)
    current_robot_config = initial_config_array.copy()  # Initialize global config
    urdf_vis.update_cfg(initial_config_array)
    
    target_link_names = ["right_end_effector_link", "left_end_effector_link"]
    link_names = list(urdf.link_map.keys())
    right_palm_idx = link_names.index("right_end_effector_link")
    left_palm_idx = link_names.index("left_end_effector_link")
    
    # Print parent-child relationships for target links
    for target_link in target_link_names:
        if target_link in urdf.link_map:
            link = urdf.link_map[target_link]
            # Find the joint that connects to this link
            parent_joint_name = None
            for joint_name, joint in urdf.joint_map.items():
                if joint.child == target_link:
                    parent_joint_name = joint_name
                    break
            if parent_joint_name:
                print(f"  {target_link} -> parent joint: {parent_joint_name}")
            else:
                print(f"  {target_link} -> no parent joint (root link)")
    
    # Get initial position of end effectors
    right_palm_transform = robot.forward_kinematics(initial_config_array, right_palm_idx)
    left_palm_transform = robot.forward_kinematics(initial_config_array, left_palm_idx)
    right_palm_pos = np.array(right_palm_transform[right_palm_idx, -3:])
    left_palm_pos = np.array(left_palm_transform[left_palm_idx, -3:])
    right_palm_quat = (1.0, 0.0, 0.0, 0.0)
    left_palm_quat = (1.0, 0.0, 0.0, 0.0)
     
    ik_target_0 = server.scene.add_transform_controls(
        "/ik_target_0", scale=0.2, position=right_palm_pos, wxyz=right_palm_quat
    )
    ik_target_1 = server.scene.add_transform_controls(
        "/ik_target_1", scale=0.2, position=left_palm_pos, wxyz=left_palm_quat
    )
    
    # Add some GUI controls
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    
    # Add gripper control sliders
    left_gripper_slider = server.gui.add_slider("Left Gripper", 0.0, 1.0, 0.05, 0.0)
    right_gripper_slider = server.gui.add_slider("Right Gripper", 0.0, 1.0, 0.05, 0.0)
 
    
    # Define joint indices - include waist for proper IK
    lower_body_indices = list(range(12))  # 0-11: legs and hips
    waist_indices = [12, 13, 14]  # waist_yaw_joint, waist_pitch_joint, waist_roll_joint
    upper_body_indices = list(range(15, len(initial_config)))  # 15+: arms and hands
    
    # Define hand gripper joint indices
    left_gripper_indices = list(range(22, 22+6))  # Left hand joints (indices 22-32)
    right_gripper_indices = list(range(35, 35+6))  # Right hand joints (indices 39-49)
     
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask_server, args=(flask_port,), daemon=True)
    flask_thread.start()
    print(f"Flask server started on port {flask_port}")
    print(f"Robot config available at: http://localhost:{flask_port}/config")
    
    current_config = initial_config_array.copy()
    while True:
        # Solve IK for both targets
        start_time = time.time()
        try:
            solution = solve_ik_with_multiple_targets(
                robot=robot,
                target_link_names=target_link_names,
                target_positions=np.array([ik_target_0.position, ik_target_1.position]),
                target_wxyzs=np.array([ik_target_0.wxyz, ik_target_1.wxyz]),
            )
            
            # Body
            current_config[lower_body_indices] = solution[lower_body_indices]
            current_config[waist_indices] = solution[waist_indices]
            current_config[upper_body_indices] = solution[upper_body_indices]
            # Left Hand
            current_config[left_gripper_indices[0:]] = -left_gripper_slider.value
            current_config[left_gripper_indices[1:]] = left_gripper_slider.value
            current_config[left_gripper_indices[2:]] = -left_gripper_slider.value
            # Right Hand
            current_config[right_gripper_indices[0:]] = -right_gripper_slider.value
            current_config[right_gripper_indices[1:]] = right_gripper_slider.value
            current_config[right_gripper_indices[2:]] = -right_gripper_slider.value
            # current_config[right_gripper_indices] = right_gripper_slider.value
            
            current_robot_config = current_config.copy()
            
            urdf_vis.update_cfg(current_config)
            
        except Exception as e:
            print(f"IK solver failed: {e}")
        
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
        
        time.sleep(0.05)


if __name__ == "__main__":
    tyro.cli(main)