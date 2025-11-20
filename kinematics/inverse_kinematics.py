'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity, matrix, matmul
from numpy import sin, cos, pi, asin, acos, atan2, atan, sqrt
from numpy.linalg import inv

class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        joint_angles = [None] * 6
        # YOUR CODE HERE
        # This code follows the analytical method described in BHuman Code Release 2017
        # (https://github.com/bhuman/BHumanCodeRelease/blob/coderelease2017/CodeRelease2017.pdf)

        # Define rotation matrices
        def Rx(theta):
            s = sin(theta)
            c = cos(theta)
            return matrix([[1,  0,  0,  0],
                           [0,  c, -s,  0],
                           [0,  s,  c,  0],
                           [0,  0,  0,  1]])
        def Ry(theta):
            s = sin(theta)
            c = cos(theta)
            return matrix([[ c,  0,  s,  0],
                           [ 0,  1,  0,  0],
                           [-s,  0,  c,  0],
                           [ 0,  0,  0,  1]])
        def Rz(theta):
            s = sin(theta)
            c = cos(theta)
            return matrix([[ c,  s,  0,  0],
                           [-s,  c,  0,  0],
                           [ 0,  0,  1,  0],
                           [ 0,  0,  0,  1]])
        
        # Leg lengths in metres
        l_hip_y = 0.1
        l_hip_z = 0.085
        l_up = 0.1
        l_low = 0.1029 + 0.04519

        # Creating HipOrth2Foot where transform = Torso2Foot
        Foot2Torso = inv(transform)
        if effector_name == 'LLeg':
            Foot2Torso[3,1] += (l_hip_y/2)
        else:
            Foot2Torso[3,1] -= (l_hip_y/2)
        Foot2Torso[3,2] -= l_hip_z
        Foot2HipOrth = matmul(Rx(pi/4), Foot2Torso)
        HipOrth2Foot = inv(Foot2HipOrth)

        # Calculate l_trans
        a = HipOrth2Foot[3,0]
        b = HipOrth2Foot[3,1]   
        c = HipOrth2Foot[3,2]
        l_trans = sqrt(a**2 + b**2 + c**2)

        # Calculate knee pitch angle
        knee_pitch_arg = (l_up**2 + l_low**2 - l_trans**2)/(2 * l_up * l_low)
        knee_pitch_arg = max(-1.0, min(1.0, knee_pitch_arg))
        knee_pitch_angle =  pi - acos(knee_pitch_arg)
        joint_angles[3] = knee_pitch_angle

        # Calculate ankle/foot pitch angle
        ankle_pitch_arg = (l_low**2 + l_trans**2 - l_up**2)/(2 * l_low * l_trans)
        ankle_pitch_arg = max(-1.0, min(1.0, ankle_pitch_arg))
        ankle_pitch1 = acos(ankle_pitch_arg)
        x = Foot2HipOrth[3,0]
        y = Foot2HipOrth[3,1]
        z = Foot2HipOrth[3,2]
        ankle_pitch2 = atan2(x, sqrt(y**2 + z**2))
        ankle_pitch_angle = ankle_pitch1 + ankle_pitch2
        joint_angles[4] = ankle_pitch_angle

        # Calculate ankle/foot roll angle
        ankle_roll_angle = atan2(y,z)
        joint_angles[5] = ankle_roll_angle

        # Thigh matrix
        m1 = matmul(Rx(ankle_roll_angle), Ry(ankle_pitch_angle))
        m1[3,2] -= l_low
        m2 = matmul(m1, Ry(knee_pitch_angle))
        m2[3,2] -= l_up
        Thigh2Foot = m2
        HipOrth2Thigh = matmul(inv(Thigh2Foot), HipOrth2Foot)
        thigh = HipOrth2Thigh

        # Calculate hip angles (yaw = yawpitch)
        hip_yaw_angle = atan2(-thigh[0,1], thigh[1,1])
        hip_roll_angle = asin(thigh[2,1]) - pi/4
        hip_pitch_angle = atan2(-thigh[2,0], thigh[2,2])
        
        joint_angles[0] = hip_yaw_angle
        joint_angles[1] = hip_roll_angle
        joint_angles[2] = hip_pitch_angle
        
        return joint_angles

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        joint_names = self.chains[effector_name]
        joint_times = [[0.0, 0.1] for i in joint_names]
        joint_angles = self.inverse_kinematics(effector_name, transform)
        keys = []
        for angle in joint_angles:
            # handle1 and handle2: [InterpolationType, dTime, dAngle]
            # InterpolationType: 0 = linear, 1 = cubic
            angle = float(angle)
            handle1 = [0, 0.0, 0.0]  # previous handle
            handle2 = [0, 0.0, 0.0]  # next handle
            keys.append([[angle, handle1, handle2], [angle, handle1, handle2]])
        print("keys:", keys)

        self.keyframes = (joint_names, joint_times, keys)  # the result joint angles have to fill in

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26
    agent.set_transforms('LLeg', T)
    agent.run()
