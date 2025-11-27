'''In this file you need to implement remote procedure call (RPC) client

* The agent_server.py has to be implemented first (at least one function is implemented and exported)
* Please implement functions in ClientAgent first, which should request remote call directly
* The PostHandler can be implement in the last step, it provides non-blocking functions, e.g. agent.post.execute_keyframes
 * Hints: [threading](https://docs.python.org/2/library/threading.html) may be needed for monitoring if the task is done
'''

import weakref, requests, threading

class PostHandler(object):
    '''the post hander wraps function to be executed in parallel
    '''
    def __init__(self, obj):
        self.proxy = weakref.proxy(obj)

    def execute_keyframes(self, keyframes):
        '''non-blocking call of ClientAgent.execute_keyframes'''
        # YOUR CODE HERE
        threading.Thread(target=self.proxy.execute_keyframes, args=(keyframes,), daemon=False).start()

    def set_transform(self, effector_name, transform):
        '''non-blocking call of ClientAgent.set_transform'''
        # YOUR CODE HERE
        threading.Thread(target=self.proxy.set_transform, args=(effector_name, transform), daemon=False).start()


class ClientAgent(object):
    '''ClientAgent request RPC service from remote server
    '''
    # YOUR CODE HERE
    def __init__(self, host="localhost", port=5000):
        self.post = PostHandler(self)
        self.url = f"http://{host}:{port}/"
        self.request_id = 0
    
    def rpc_call(self, method, params):
        self.request_id += 1
        payload = {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params or {},
                    "id": self.request_id
                  }

        response = requests.post(self.url, json=payload)
        response_data = response.json()

        if "error" in response_data:
            raise RuntimeError(f"RPC Error: {response_data}")
        print(response_data)
        return response_data

    def get_angle(self, joint_name):
        '''get sensor value of given joint'''
        # YOUR CODE HERE
        return self.rpc_call("get_angle", {"joint_name": joint_name})
    
    def set_angle(self, joint_name, angle):
        '''set target angle of joint for PID controller
        '''
        # YOUR CODE HERE
        return self.rpc_call("set_angle", {"joint_name": joint_name, "angle": angle})

    def get_posture(self):
        '''return current posture of robot'''
        # YOUR CODE HERE
        return self.rpc_call("get_posture", {})

    def execute_keyframes(self, keyframes):
        '''execute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        '''
        # YOUR CODE HERE
        return self.rpc_call("execute_keyframes", {"keyframes": keyframes})

    def get_transform(self, joint_name):
        '''get transform with given name
        '''
        # YOUR CODE HERE
        return self.rpc_call("get_transform", {"joint_name": joint_name})

    def set_transform(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        return self.rpc_call("set_transform", {"effector_name": effector_name, "transform": transform})

if __name__ == '__main__':
    agent = ClientAgent()
    # TEST CODE HERE
    # agent.get_angle("HeadPitch")
    # agent.set_angle("HeadYaw", 1.0)
    # agent.get_posture()
    # agent.get_transform("LKneePitch")
    # agent.execute_keyframes("leftBackToStand")
    # agent.post.execute_keyframes("wipe_forehead")
    # agent.post.set_transform("LLeg", [0, 0.05, -0.26])
