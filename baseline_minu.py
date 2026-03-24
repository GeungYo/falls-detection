import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import math

AXIS = 1
HUMANCNT = 1
JOINTCNT = 25
CRITICAL = 11 #점 몇개가 범위 안에 있어야 할 지
NEEDJOINT = [4, 9, 21, 5, 10, 11, 6, 7, 2, 17, 1, 13, 18, 19, 15, 16]
filePath = "fall.skeleton"


def getCoor(): # 2차원 리스트 [프레임][조인트], 프레임 수 반환
    file = open(filePath, 'r')

    frameCnt = int(file.readline().strip()) #맨 첫줄 프레임 수

    coordinate = [] #전체 프레임 좌표 / 2차원 리스트 [프레임][조인트]
    oneFrameCoor = [0] #프레임 하나 좌표

    while True :
        nowLine = file.readline()
        
        if nowLine == '': 
            break #파일 끝이니깐 종료

        stripTmp = nowLine.strip()
        if stripTmp == str(HUMANCNT):
            file.readline() #사람 카운트 다음 줄 bodyID 인데 안쓰니깐 버려주기
            continue
        if stripTmp == str(JOINTCNT):
            continue

        #이후로는 좌표값들
        #print(stripTmp)
        listTmp = list(map(float, stripTmp.split(' ')))
        oneFrameCoor.append(listTmp[AXIS])

        if len(oneFrameCoor) == JOINTCNT: #조인트 개수만큼 다 채웠으면 한 프레임 끝
            coordinate.append(oneFrameCoor)
            oneFrameCoor = [0]

    file.close()
    return coordinate, frameCnt

def getJoint_17_18():
    file = open(filePath, 'r')
    lines = file.readlines()

    idx = 1
    num = int(lines[idx].strip())
    idx += 1

    for _ in range(num):
        idx += 1 #BodyID 스킵

        numJoints = int(lines[idx].strip())
        idx += 1

        joints = []

        for _ in range(numJoints):
            jointInfo = list(map(float, lines[idx].strip().split()))
            joints.append(jointInfo[0:3]) # [x, y, z]
            idx += 1

        if len(joints) >= 18:
            file.close()
            return joints[16], joints[17] # 17, 18번

    file.close()
    return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]


            
def isFall(oneFrameCoor, critPoint1, critPoint2) : #매개변수는 프레임 하나(1차원 리스트), 임계점 / 누운 상태 True, 아님 False
    pointCnt = 0
    for i in NEEDJOINT:
        if critPoint1 <= oneFrameCoor[i] and oneFrameCoor[i] <= critPoint2 :
            pointCnt+=1

    if pointCnt >= CRITICAL:
        return True
    else:
        return False

arr, frameCnt = getCoor()

#길이 계산
j17, j18 = getJoint_17_18()
lenThigh = 0
for i in range(3):
    lenThigh += (abs(j17[i] - j18[i])) ** 2
lenThigh = math.sqrt(lenThigh)

res = [False]
for i in range(frameCnt):
    lower = min(arr[i]) # 가장 아래 값 
    res.append(isFall(arr[i], lower, lower + lenThigh))

# ---------------------------
# 1. skeleton 파일 파싱
# ---------------------------
def load_skeleton(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    idx = 0
    num_frames = int(lines[idx].strip())
    idx += 1

    data = []

    for _ in range(num_frames):
        num_bodies = int(lines[idx].strip())
        idx += 1

        frame_data = []

        for _ in range(num_bodies):
            idx += 1  # body info skip

            num_joints = int(lines[idx].strip())
            idx += 1

            joints = []
            for _ in range(num_joints):
                joint_info = list(map(float, lines[idx].strip().split()))
                x, y, z = joint_info[0:3]
                joints.append([x, y, z])
                idx += 1

            frame_data.append(joints)

        # 사람 여러 명이면 첫 번째만 사용
        if len(frame_data) > 0:
            data.append(frame_data[0])
        else:
            data.append(np.zeros((25, 3)))

    return np.array(data)  # shape: (T, 25, 3)


# ---------------------------
# 2. 관절 연결 정의
# ---------------------------
# NTU skeleton bone 연결
bones = [
    (0,1),(1,20),(20,2),(2,3),
    (20,4),(4,5),(5,6),(6,7),
    (20,8),(8,9),(9,10),(10,11),
    (0,12),(12,13),(13,14),(14,15),
    (0,16),(16,17),(17,18),(18,19)
]


# ---------------------------
# 3. 시각화
# ---------------------------
def visualize_skeleton(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()

        joints = data[frame]
        xs, ys, zs = joints[:,0], joints[:,1], joints[:,2]

        # 관절 점
        ax.scatter(xs, ys, zs)

        # 뼈 연결
        for b in bones:
            x = [xs[b[0]], xs[b[1]]]
            y = [ys[b[0]], ys[b[1]]]
            z = [zs[b[0]], zs[b[1]]]
            ax.plot(x, y, z)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 2)
        ax.set_zlim(0, 10)

        ax.set_title(f"Frame {frame} / Fall : {res[frame]}")

    ani = FuncAnimation(fig, update, frames=len(data), interval=50)
    plt.show()


# ---------------------------
# 실행
# ---------------------------
file_path = "fall.skeleton"
data = load_skeleton(file_path)

print("shape:", data.shape)  # (T, 25, 3)

visualize_skeleton(data)
