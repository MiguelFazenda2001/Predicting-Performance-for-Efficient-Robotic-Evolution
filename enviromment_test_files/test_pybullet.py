import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version which is lighter 
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf") # plane é o plano para o robo ficar e nao continuar a cair
huskypos = [0, 0, 0.1]
ori = [0, 0, 0.5, 0]

roboID = p.loadURDF("husky/husky.urdf", [0, 0, 0.5], ori,) # todos estas variaveis vao ser da posição

movable_joints = []

numJoints = p.getNumJoints(roboID)
print(f"Number of joints here --> {numJoints}")
for joint in range(numJoints):
    print(f"Informações de cada joint {p.getJointInfo(roboID,joint)[0]} --> {p.getJointInfo(roboID,joint)[1:]}") 
    if p.getJointInfo(roboID, joint)[2] in [0]:
        movable_joints.append(joint)


# Moving with torque control
# Set target velocity to 0
# and maximum force for the wheels

targetVel = 0 # aqui em rad/s, aqui é a valocidade, negativo vai pra tras e positivo pra frente
maxForce = -10 # aqui em newton

"""
for step in range(100000):
    p.stepSimulation()

    joint_state = p.getJointState(roboID, 2)
    joint_velocity = joint_state[1]  # Index 1 is the joint velocity
    print(f"Step {step}, Joint {2}, Velocity: {joint_velocity:.4f}")

    time.sleep(1.0 / 240.0)
"""
print("STOP HERE")
for _ in range(600):
    for joint in movable_joints:
        p.setJointMotorControl2(roboID, joint, controlMode=p.VELOCITY_CONTROL, force=0)
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

for step in range(600):
    # Aplica o torque manualmente nas 4 rodas
    for joint in movable_joints:
        p.setJointMotorControl2(roboID, joint, controlMode=p.TORQUE_CONTROL, force=maxForce)
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

p.disconnect()