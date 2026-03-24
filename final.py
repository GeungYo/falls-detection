import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import math

# --- 권민우 설정 ---
AXIS = 1
HUMANCNT = 1
JOINTCNT = 25
CRITICAL = 11
NEEDJOINT = [4, 9, 21, 5, 10, 11, 6, 7, 2, 17, 1, 13, 18, 19, 15, 16]
filePath = "fall.skeleton"
FPS = 30

target_idx = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]
connection = [
    (0, 1), (1, 16), (16, 2), (2, 3),
    (16, 4), (4, 5), (5, 6),
    (16, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15)
]

def getCoor():  # 2차원 리스트 [프레임][조인트], 프레임 수 반환
    file = open(filePath, 'r')

    frameCnt = int(file.readline().strip())  # 맨 첫줄 프레임 수

    coordinate = []  # 전체 프레임 좌표 / 2차원 리스트 [프레임][조인트]
    oneFrameCoor = []  # 프레임 하나 좌표

    while True:
        nowLine = file.readline()

        if nowLine == '':
            break  # 파일 끝이니깐 종료

        stripTmp = nowLine.strip()
        if stripTmp == str(HUMANCNT):
            file.readline()  # 사람 카운트 다음 줄 bodyID 인데 안쓰니깐 버려주기
            continue
        if stripTmp == str(JOINTCNT):
            continue

        listTmp = list(map(float, stripTmp.split()))
        oneFrameCoor.append(listTmp[AXIS])

        if len(oneFrameCoor) == JOINTCNT:  # 조인트 개수만큼 다 채웠으면 한 프레임 끝
            coordinate.append(oneFrameCoor)
            oneFrameCoor = []

    file.close()
    return coordinate, frameCnt

def getJoint_17_18():
    file = open(filePath, 'r')
    lines = file.readlines()

    idx = 1
    num = int(lines[idx].strip())
    idx += 1

    for _ in range(num):
        idx += 1  # BodyID 스킵

        numJoints = int(lines[idx].strip())
        idx += 1

        joints = []

        for _ in range(numJoints):
            jointInfo = list(map(float, lines[idx].strip().split()))
            joints.append(jointInfo[0:3])  # [x, y, z]
            idx += 1

        if len(joints) >= 18:
            file.close()
            return joints[16], joints[17]  # 17, 18번

    file.close()
    return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

def isFall(oneFrameCoor, critPoint1, critPoint2):
    pointCnt = 0
    for i in NEEDJOINT:
        if critPoint1 <= oneFrameCoor[i] and oneFrameCoor[i] <= critPoint2:
            pointCnt += 1
    return pointCnt >= CRITICAL

# ---------------------------
# 전체 영상 기준 낙상 판단
# ---------------------------
arr, frameCnt = getCoor()

j17, j18 = getJoint_17_18()
lenThigh = 0.0
for i in range(3):
    lenThigh += (abs(j17[i] - j18[i])) ** 2
lenThigh = math.sqrt(lenThigh)

fall_detected = False

for i in range(FPS, frameCnt):
    head_prev = arr[i - FPS][3]
    head_now = arr[i][3]

    if head_prev != 0 and (head_prev - head_now) / head_prev >= 0.3:
        lower = min(arr[i])
        if isFall(arr[i], lower, lower + lenThigh):
            fall_detected = True
            break

# ---------------------------
# 시각화용 데이터 로드
# ---------------------------
def load_skeleton(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    num_f = int(lines[0].strip())
    data, idx = [], 1

    for _ in range(num_f):
        nb = int(lines[idx].strip())
        idx += 1

        if nb > 0:
            idx += 1
            nj = int(lines[idx].strip())
            idx += 1
            joints = [list(map(float, lines[idx + i].split()[:3])) for i in range(nj)]
            data.append(np.array(joints))
            idx += nj
        else:
            data.append(np.zeros((25, 3)))

    return np.array(data)

frames_data = load_skeleton(filePath)

# ---------------------------
# 시각화
# ---------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 네 설정 그대로 유지
ax.set(
    xlim=(-1, 1),
    ylim=(0, 5),
    zlim=(0, 2),
    xlabel='X',
    ylabel='Depth(Z)',
    zlabel='Height(Y)'
)

lines_3d = [ax.plot([], [], [], 'b-')[0] for _ in connection]
scatter = ax.scatter([], [], [], c='r', s=2)

def update(frame):
    pts = frames_data[frame][target_idx]

    scatter._offsets3d = (pts[:, 0], pts[:, 2], pts[:, 1])

    for line, (j1, j2) in zip(lines_3d, connection):
        line.set_data_3d(
            [pts[j1, 0], pts[j2, 0]],
            [pts[j1, 2], pts[j2, 2]],
            [pts[j1, 1], pts[j2, 1]]
        )

    ax.set_title(f"Frame {frame} / Fall: {fall_detected}")
    return lines_3d + [scatter]

ani = FuncAnimation(fig, update, frames=len(frames_data), interval=30, blit=False)
plt.show()