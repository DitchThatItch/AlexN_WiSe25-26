'''In this file you need to implement remote procedure call (RPC) server

* There are different RPC libraries for python, such as xmlrpclib, json-rpc. You are free to choose.
* The following functions have to be implemented and exported:
 * get_angle
 * set_angle
 * get_posture
 * execute_keyframes
 * get_transform
 * set_transform
* You can test RPC server with ipython before implementing agent_client.py
'''

# add PYTHONPATH
import os
import sys
import json, threading, time
from jsonrpc import JSONRPCResponseManager, dispatcher
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from numpy.matlib import matrix, identity

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'kinematics'))
from inverse_kinematics import InverseKinematicsAgent

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))
from keyframes import hello, leftBackToStand, leftBellyToStand, rightBackToStand, rightBellyToStand, wipe_forehead

class ServerAgent(InverseKinematicsAgent):
    '''ServerAgent provides RPC service
    '''
    # YOUR CODE HERE
    def __init__(self):
        super().__init__()
        # Register all RPC methods
        dispatcher.add_method(self.get_angle,          "get_angle")
        dispatcher.add_method(self.set_angle,          "set_angle")
        dispatcher.add_method(self.get_posture,        "get_posture")
        dispatcher.add_method(self.execute_keyframes,  "execute_keyframes")
        dispatcher.add_method(self.get_transform,      "get_transform")
        dispatcher.add_method(self.set_transform,      "set_transform")

        self.keyframe_list = [
                              "hello", 
                              "leftBackToStand", 
                              "leftBellyToStand", 
                              "rightBackToStand", 
                              "rightBellyToStand", 
                              "wipe_forehead"
                              ]

    class RequestHandler(BaseHTTPRequestHandler):
        
        def do_POST(self):
            # Read request data
            content_length = int(self.headers.get('Content-Length', 0))
            request_json = self.rfile.read(content_length).decode()

            # Handle the JSON-RPC request
            response = JSONRPCResponseManager.handle(request_json, dispatcher)

            # Send the response
            response_json = json.dumps(response.json)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json.encode())
    
    def run_server(self):
        host="0.0.0.0"
        port=5000
        print(f"JSON-RPC server running at http://{host}:{port}/")
        server = ThreadingHTTPServer((host, port), self.RequestHandler)
        server.serve_forever()
    
    def run(self):
        super().run()

    def get_angle(self, joint_name):
        '''get sensor value of given joint'''
        # YOUR CODE HERE
        print("Called get_angle")
        if joint_name not in self.transforms:
            raise Exception(f"ERROR: Unknown joint: {joint_name}")
        angle = self.perception.joint[joint_name]
        return angle
    
    def set_angle(self, joint_name, angle):
        '''set target angle of joint for PID controller
        '''
        # YOUR CODE HERE
        print("Called set_angle")
        if joint_name not in self.transforms:
            raise Exception(f"ERROR: Unknown joint: {joint_name}")
        self.target_joints[joint_name] = angle
        return "Success"

    def get_posture(self):
        '''return current posture of robot'''
        # YOUR CODE HERE
        print("Called get_posture")
        posture = float(self.posture)
        return posture

    def execute_keyframes(self, keyframes):
        '''execute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        '''
        # YOUR CODE HERE                                                
        print("Called execute_keyframes")
        
        if keyframes not in self.keyframe_list:
            raise Exception(f"ERROR: Unknown keyframes: {keyframes}")
        
        if keyframes == "hello":
            self.keyframes = hello()
        if keyframes == "leftBackToStand":
            self.keyframes = leftBackToStand()
        if keyframes == "leftBellyToStand":
            self.keyframes = leftBellyToStand()
        if keyframes == "rightBackToStand":
            self.keyframes = rightBackToStand()
        if keyframes == "rightBellyToStand":
            self.keyframes = rightBellyToStand()
        if keyframes == "wipe_forehead":
            self.keyframes = wipe_forehead(motion=None)
        
        end_time = self.keyframes[1][0][-1]
        print(end_time)

        self.motion_start_time = self.perception.time
        
        time.sleep(0.05)    # allow angle_interpolation to set t_now before checking
        while True:

            if self.t_now >= end_time:
                break
            
            time.sleep(0.01)  # 10ms sleep to avoid busy wait

        return "Success"

    def get_transform(self, joint_name):
        '''get transform with given name
        '''
        # YOUR CODE HERE
        print("Called get_transform")
        if joint_name not in self.transforms:
            raise Exception(f"ERROR: Unknown joint: {joint_name}")
    
        angle = self.perception.joint[joint_name]

        T = self.local_trans(joint_name, angle)
        
        return T.tolist()
    
    def set_transform(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        print("Called set_transform")
        T = identity(4)
        
        if effector_name not in self.chains:
            raise Exception(f"ERROR: Unknown effector: {effector_name}")
        if len(transform) != 3:
            raise Exception("ERROR: Input must be [x,y,z] list")
        
        T[3, 0:3] = transform
        
        self.set_transforms(effector_name, T)
        return "Success"

if __name__ == '__main__':
    agent = ServerAgent()
    threading.Thread(target=agent.run, daemon=True).start()
    agent.run_server()
