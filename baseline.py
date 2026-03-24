import numpy as np

def check_fall(frames, thud_idx, fps=30):
    # 임계값 설정
    h_th = 0     # 바닥 판단 높이
    drop_th = 0   # 낙하 폭
    var_th = 0   # 움직임(분산) 허용치
    fps2 = fps * 2  # 2초 프레임 수

    # 데이터 길이 예외 처리
    if thud_idx < fps or thud_idx + fps2 >= len(frames):
        return False

    # 쿵 소리 직전 1초간 확 떨어졌는지 (머리 관절 3번)
    head_drop = frames[thud_idx - fps, 3, 1] - frames[thud_idx, 3, 1]
    if head_drop < drop_th:
        return False

    # 바닥 근접 좌표 10개 이상인지 (머리는 기본 1점 + 추가 2점 = 총 3점)
    y_vals = frames[thud_idx, :, 1]
    score = sum(1 for y in y_vals if y < h_th)
    if y_vals[3] < h_th:
        score += 2  
        
    if score < 10:
        return False

    # 3 & 4. 2초 동안 안 일어나는지 & 움직임이 없는지
    post_y = frames[thud_idx : thud_idx + fps2, :, 1]
    
    if np.mean(post_y[:, 3]) > h_th: # 일어났는지 확인
        return False
        
    if np.var(post_y) > var_th:      # 큰 움직임이 있는지 확인
        return False

    return True