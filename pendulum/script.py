import gymnasium as gym
import numpy as np


def pd_controller(state, kp=100.0, kd=10.0):
    theta = np.arctan2(state[1], state[0]) 
    theta_dot = state[2]  

    desired_theta = 0.0
    error = theta - desired_theta
    torque = -kp * error - kd * theta_dot
    return np.clip(torque, -2.0, 2.0)

def energy_controller(state, kp=5.0, kd=1.0):
    theta = np.arctan2(state[1], state[0]) 
    theta_dot = state[2]  
    m=1
    l=1
    g=9.81

    energy = 0.5 * (theta_dot**2) + m * g * l * (np.cos(theta)-1)    
    print(energy)

    if abs(energy)<0.1:
        control_mode=0
    else:
        control_mode=1

    desired_energy = 0.0
    error = energy - desired_energy

    if energy>desired_energy:
        torque = kp * error - kd * theta_dot
    else:
        torque = -0.1*error * theta_dot

    return np.clip(torque, -2.0, 2.0), control_mode

env = gym.make("Pendulum-v1",render_mode="human",g=9.81)

state, info = env.reset(seed=42)

#set the initial state
env.unwrapped.state = np.array([3.14, 0.5])
obs = env.unwrapped._get_obs()
next_state=[0,0,0]
control_mode=1

for _ in range(2000):
    env.render()

    if control_mode==0:
        action= pd_controller(state)
    else:
        action, control_mode= energy_controller(state)

    state, reward, terminated, truncated, info = env.step([action])

    if terminated or truncated:
        break

env.close()
