'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


from pid import PIDAgent
from keyframes import leftBackToStand


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        # Some keyframes dont have LHipYawPitch!
        target_joints['RHipYawPitch'] = target_joints['LHipYawPitch'] # copy missing joint in keyframes
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}
        # YOUR CODE HERE
        if not hasattr(self, 'motion_start_time'):
            self.motion_start_time = perception.time

        t_now = perception.time - self.motion_start_time
        names = keyframes[0]

        for j in range(len(names)): 

            times = keyframes[1][j]
            keys = keyframes[2][j]

            for i in range(len(times) - 1):
                if times[i] <= t_now <= times[i+1]:
                    break
            else:
                if t_now < times[0]:
                    i = 0
                else:
                    i = len(times) - 2
            
            ang_start = keys[i][0]
            ang_end = keys[i+1][0]
            handle1 = keys[i][2]
            handle2 = keys[i+1][1]

            P0 = ang_start
            P1 = ang_start + handle2[2]
            P2 = ang_end + handle1[2]
            P3 = ang_end

            t0, t1 = times[i], times[i+1]
            u = (t_now - t0) / (t1 - t0)
            u = max(0.0, min(1.0, u))

            angle = (1-u)**3*P0 + 3*(1-u)**2*u*P1 + 3*(1-u)*u**2*P2 + u**3*P3
            target_joints[names[j]] = angle

        return target_joints

if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = leftBackToStand()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
