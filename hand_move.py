# python3 hand_move.py --port 8000 --flask-port 5000
# curl http://localhost:5000/config

import time
import tyro
import viser
import numpy as np

from flask import Flask, jsonify
import threading


# Global variables to store target poses
target_poses = {
    "end_effector": {"position": None, "orientation": None, "gripper": None}
}

# Create Flask app
app = Flask(__name__)

@app.route('/target_pose', methods=['GET'])
def get_target_poses():
    """Return the current target poses as JSON"""
    global target_poses
    
    return jsonify({
        "target_poses": target_poses,
        "timestamp": time.time()
    })

def run_flask_server(flask_port=5000):
    """Run Flask server in a separate thread"""
    app.run(host='0.0.0.0', port=flask_port, debug=False, use_reloader=False)

def main(port: int, flask_port: int = 5000) -> None:
    global target_poses
    
    server = viser.ViserServer(port=port)
    server.scene.add_grid("/ground", width=2, height=2)
    
    # Set initial position for the IK target
    palm_pos = np.array([0.0, 0.0, 0.0])  # Hand initial position
    palm_quat = (1.0, 0.0, 0.0, 0.0)  # Identity quaternion
    
    # Create IK target control
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=palm_pos, wxyz=palm_quat
    )
    
    # Add some GUI controls
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    
    # Add gripper control slider
    gripper_slider = server.gui.add_slider("Gripper", 0.0, 1.0, 0.01, 0.0)
    
    # Add reset button
    reset_button = server.gui.add_button("Reset to Initial Pose")
    
    # Store initial pose for reset functionality
    initial_palm_pos = palm_pos.copy()
    initial_palm_quat = palm_quat
    
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask_server, args=(flask_port,), daemon=True)
    flask_thread.start()
    print(f"Flask server started on port {flask_port}")
    print(f"Target poses available at: http://localhost:{flask_port}/target_pose")
    
    while True:
        # Check if reset button was pressed
        if reset_button.value:
            # Reset IK target to initial pose
            ik_target.position = initial_palm_pos
            ik_target.wxyz = initial_palm_quat
            reset_button.value = False  # Reset the button state
            print("Reset to initial pose")
        
        # Update target poses from the transform controls
        target_poses["end_effector"]["position"] = ik_target.position.tolist()
        target_poses["end_effector"]["orientation"] = ik_target.wxyz.tolist()
        target_poses["end_effector"]["gripper"] = gripper_slider.value
        
        # Update timing
        elapsed_time = 0.05  # Fixed time step
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
        
        time.sleep(0.05)


if __name__ == "__main__":
    tyro.cli(main)