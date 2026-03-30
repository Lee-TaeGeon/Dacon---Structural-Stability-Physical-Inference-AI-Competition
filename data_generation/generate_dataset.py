import pybullet as pb
import pybullet_data
import numpy as np
import cv2
import os
import pandas as pd
import shutil
import json

# ==========================================
# --- 1. 설정 (CFG) ---
# ==========================================
IMG_SIZE = 384
SEED = 42
np.random.seed(SEED)

NUM_SAMPLES = 3000  # 1차 목표를 위해 충분한 양 설정
OUTPUT_DIR = os.path.join(os.getcwd(), "newLabel_Dataset")
BLOCK_X = 0.2    # 가로(x)
BLOCK_Y = 0.2    # 세로(y)
BLOCK_Z = 0.15   # 높이(z)

BLOCK_MASS = 0.34
THRESHOLD_DIST = 0.015

PASTEL_COLORS = [
    [0.58, 0.57, 0.44, 1.0],  # 올리브 그레이
    [0.62, 0.46, 0.50, 1.0],  # 보라빛 회갈색
    [0.66, 0.49, 0.44, 1.0],  # 더스티 로즈
    [0.45, 0.58, 0.43, 1.0],  # 탁한 연두
    [0.51, 0.50, 0.47, 1.0],  # 웜 그레이
    [0.44, 0.47, 0.56, 1.0],  # 슬레이트 블루
    [0.68, 0.63, 0.53, 1.0],  # 베이지 브라운
]

# ==========================================
# --- 2. 유틸리티 함수 ---
# ==========================================

def setup_directories():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_pastel_color():
    base = np.random.uniform(0.4, 0.8, size=3)
    gray_mix = np.random.uniform(0.2, 0.5)
    pastel = base * (1 - gray_mix) + gray_mix
    
    # 🔥 대비 강화 (중요)
    contrast = np.random.uniform(0.85, 1.15)
    pastel = np.clip(pastel * contrast, 0, 1)
    
    return list(pastel) + [1.0]

def create_block(pos, color):
    half_extents = [BLOCK_X/2, BLOCK_Y/2, BLOCK_Z/2]
    col_shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half_extents)
    vis_shape = pb.createVisualShape(pb.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    rand_yaw = np.random.uniform(-0.02, 0.02) 
    orn = pb.getQuaternionFromEuler([0, 0, rand_yaw])
    
    block_id = pb.createMultiBody(baseMass=BLOCK_MASS, 
                                  baseCollisionShapeIndex=col_shape, 
                                  baseVisualShapeIndex=vis_shape, 
                                  basePosition=pos,
                                  baseOrientation=orn) # 회전 적용
    pb.changeVisualShape(
        block_id, -1,
        specularColor=[0.2, 0.2, 0.2],  # 반사 살짝 추가
        rgbaColor=color
)

    pb.changeDynamics(
        block_id, -1,
        lateralFriction=0.7,
        restitution=0.02,
        rollingFriction=0.02,    # 추가
        spinningFriction=0.02    # 추가
)
    
    return block_id   #  이거 추가

def render_camera_with_ranges(cam_pos, tgt_pos, fov, width=IMG_SIZE, height=IMG_SIZE):
    view_matrix = pb.computeViewMatrix(cameraEyePosition=cam_pos, cameraTargetPosition=tgt_pos, cameraUpVector=[0, 0, 1])
    proj_matrix = pb.computeProjectionMatrixFOV(fov=fov, aspect=width/height, nearVal=0.1, farVal=100.0)
    
    _, _, rgb, _, _ = pb.getCameraImage(width, height, view_matrix, proj_matrix, renderer=pb.ER_BULLET_HARDWARE_OPENGL)
    
    # uint8 변환 (OpenCV 에러 방지)
    rgb_array = np.array(rgb, dtype=np.uint8).reshape((height, width, 4))
    bgr_array = cv2.cvtColor(rgb_array[:, :, :3], cv2.COLOR_RGB2BGR)
    return bgr_array

def generate_structure_coords(structure_type):
    """구조별 좌표 생성 로직"""
    coords = []
    margin = 0.01
    # 블록 배치 시 기준이 되는 단위 (블록 크기 + 간격)
    unit_w = BLOCK_X + margin
    unit_h = BLOCK_Z  # 높이는 층간 유격을 최소화해야 안정적임
    
    if structure_type == 'tower':
        h = np.random.randint(9, 16)
        curr_x, curr_y = 0, 0
        drift_limit = 0.01
        for z in range(h): 
            curr_x += np.random.uniform(-drift_limit, drift_limit)
            curr_y += np.random.uniform(-drift_limit, drift_limit)
            coords.append((curr_x, curr_y, z * unit_h + BLOCK_Z/2))

    elif structure_type == 'pyramid':
        levels = np.random.randint(5, 8)
        for z in range(levels):
            w = levels - z
            off = (levels - w) * unit_w / 2
            for x in range(w):
                for y in range(w):
                    cx = x * unit_w + off - (levels * unit_w / 2)
                    cy = y * unit_w + off - (levels * unit_w / 2)
                    cz = z * unit_h + (BLOCK_Z / 2)
                    coords.append((cx, cy, cz))

    elif structure_type == 'overhang': 
        height = np.random.randint(8, 12)
        # 너비(BLOCK_X)를 기준으로 기울기 비율 설정
        lean_factor = np.random.uniform(0.2, 0.35) 
        for z in range(height):
            shift_x = z * (BLOCK_X * lean_factor)
            coords.append((shift_x, 0, z * unit_h + BLOCK_Z/2))
            
    elif structure_type == 'wall':
        width = np.random.randint(5, 9)
        height = np.random.randint(8, 16)
        for x in range(width):
            for z in range(height):
                cx = x * unit_w - (width * unit_w / 2) + (unit_w / 2)
                cz = z * unit_h + (BLOCK_Z / 2)
                coords.append((cx, 0, cz))

    elif structure_type == 'grid_tower':
        grid_size = np.random.choice([2, 3, 4])
        height = np.random.randint(8, 12)
        for z in range(height):
            for x in range(grid_size):
                for y in range(grid_size):
                    cx = (x * unit_w) - (grid_size * unit_w / 2) + (unit_w / 2)
                    cy = (y * unit_w) - (grid_size * unit_w / 2) + (unit_w / 2)
                    cz = z * unit_h + (BLOCK_Z / 2)
                    coords.append((cx, cy, cz))

    elif structure_type == 'spire':
        # 1. 전체 높이 및 단계 설정 (BLOCK_Z 기준)
        H_total = np.random.randint(9, 16)
        num_stages = np.random.randint(2, 4)
        # 단계별 너비 (5x5 -> 3x3 -> 1x1 또는 3x3 -> 1x1)
        widths = [5, 3, 1] if num_stages == 3 else [3, 1]

        # 2. 단계별 층수 분배
        if num_stages == 3:
            h1 = np.random.randint(2, 4)
            h2 = np.random.randint(2, 4)
            h3 = max(1, H_total - h1 - h2) # 최소 1층 보장
            stage_heights = [h1, h2, h3]
        else:
            h1 = np.random.randint(3, 6)
            h3 = max(1, H_total - h1)
            stage_heights = [h1, h3]

        current_layer = 0
        for stage, w in enumerate(widths):
            num_layers = stage_heights[stage]
            # 상단 1x1 스파이어인지 확인 (흔들림 노이즈용)
            is_top_spire = (w == 1)

            for l in range(num_layers):
                # 너비 차이에 따른 오프셋 계산 (전체 중심을 0,0으로 맞춤)
                # widths[0]는 가장 밑단(가장 넓은 층)의 너비
                for x in range(w):
                    for y in range(w):
                        # (x - (w-1)/2)를 통해 해당 층의 로컬 중심을 0으로 만든 뒤 배치
                        cx = (x - (w - 1) / 2.0) * unit_w
                        cy = (y - (w - 1) / 2.0) * unit_w
                        cz = current_layer * unit_h + (BLOCK_Z / 2.0)
                        
                        # 상단 1x1 부분에만 추가적인 미세 흔들림 부여 (물리적 현실감)
                        if is_top_spire and H_total > 10:
                            cx += np.random.uniform(-0.02, 0.02)
                            cy += np.random.uniform(-0.02, 0.02)

                        coords.append((cx, cy, cz))
                current_layer += 1
                
    elif structure_type == 'zigzag_tower':
        h = np.random.randint(10, 15)

        for z in range(h):
            #  0 ~ 30% 범위
            offset = BLOCK_X * np.random.uniform(0.0, 0.3)

            #  지그재그 방향 (좌우 번갈아)
            cx = offset if z % 2 == 0 else -offset

            coords.append((cx, 0, z * unit_h + BLOCK_Z/2))
        
    elif structure_type == 'leaning_grid_tower':
        grid_size = np.random.choice([2, 3]) 
        height = np.random.randint(10, 15)
        angle = np.random.uniform(0, 2 * np.pi) 
        # 매 층마다 블록 너비의 7~12%씩 밀려나도록 설정 (매우 아슬아슬함)
        lean_strength = np.random.uniform(BLOCK_X * 0.07, BLOCK_X * 0.12)
        
        for z in range(height):
            lean_offset_x = z * lean_strength * np.cos(angle)
            lean_offset_y = z * lean_strength * np.sin(angle)
            for x in range(grid_size):
                for y in range(grid_size):
                    cx = (x - (grid_size - 1) / 2.0) * unit_w + lean_offset_x
                    cy = (y - (grid_size - 1) / 2.0) * unit_w + lean_offset_y
                    cz = z * unit_h + (BLOCK_Z / 2.0)
                    coords.append((cx, cy, cz))

            
    # 미세 노이즈 (생성 시 좌표가 완전히 겹치는 것을 방지)
    # margin보다 작은 노이즈를 주어 간격 효과를 유지합니다.
    noise_level = margin * 0.95
    return [
        (
            c[0] + np.random.uniform(-noise_level, noise_level), 
            c[1] + np.random.uniform(-noise_level, noise_level), 
            c[2]
        ) for c in coords
    ]    
    
def render_camera_with_ranges(cam_x_range, cam_y_range, cam_z_range,
                              tgt_x_range, tgt_y_range, tgt_z_range,
                              fov_range, width=384, height=384):
    """지정된 [최소값, 최대값] 범위 내에서 무작위로 카메라 파라미터를 하나 뽑아 렌더링합니다."""
    
    # 1. 카메라 위치 (X, Y, Z 범위)
    cam_x = np.random.uniform(cam_x_range[0], cam_x_range[1])
    cam_y = np.random.uniform(cam_y_range[0], cam_y_range[1])
    cam_z = np.random.uniform(cam_z_range[0], cam_z_range[1])
    
    # 2. 타겟 위치 (카메라가 바라보는 중심점)
    tgt_x = np.random.uniform(tgt_x_range[0], tgt_x_range[1])
    tgt_y = np.random.uniform(tgt_y_range[0], tgt_y_range[1])
    tgt_z = np.random.uniform(tgt_z_range[0], tgt_z_range[1])
    
    # 3. 화각 (FOV 범위)
    fov = np.random.uniform(fov_range[0], fov_range[1])
    roll_offset = np.random.uniform(-0.1, 0.1)
    
    view_matrix = pb.computeViewMatrix(
    cameraEyePosition=[cam_x, cam_y, cam_z],
    cameraTargetPosition=[tgt_x, tgt_y, tgt_z],
    cameraUpVector=[0, 0, 1]   # 🔥 고정
)
    
    proj_matrix = pb.computeProjectionMatrixFOV(
        fov=fov, aspect=width/height, nearVal=0.1, farVal=100.0
    )
    
    # 광원
    lx = np.random.uniform(3.0, 5.0) * np.random.choice([-1, 1])
    ly = np.random.uniform(3.0, 5.0) * np.random.choice([-1, 1])
    lz = np.random.uniform(8.0, 12.0)  # 🔥 낮추기 (그림자 선명)

    light_dir = np.array([lx, ly, lz])
    light_dir = light_dir / np.linalg.norm(light_dir)
    light_dir = light_dir.tolist()

    # [핵심] getCameraImage 옵션 강화
    _, _, rgb, _, _ = pb.getCameraImage(
        width, height, view_matrix, proj_matrix, 
        renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        lightDirection=light_dir,           
        lightColor=[1.0, 1.0, 1.0],         # 백색광
        lightDistance=1000.0,                 # 너무 멀면 그림자가 흐려짐
        lightAmbientCoeff=0.05,   # 환경광 ↓ → 대비 증가
        lightDiffuseCoeff=0.9,    # 직접광 ↑ → 입체감
        lightSpecularCoeff=0.05,   # 약한 반사 → 경계 강조             # [추가] 반사광 계수를 0으로 설정하여 '쨍한' 반사 제거
        shadow=1,                           # 그림자 ON
        flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX # 렌더링 플래그 추가 도움
    )
    
    # 1. 리스트 형태인 rgb를 numpy 배열로 변환
    rgb_array = np.array(rgb, dtype=np.uint8) 
    
    # 2. (height, width, 4) 모양으로 재배열
    rgb_array = np.reshape(rgb_array, (height, width, 4))
    
    # 3. RGBA에서 RGB만 추출 (마지막 채널 제외)
    rgb_only = rgb_array[:, :, :3]
    
    # 4. RGB를 BGR로 변환 (OpenCV 저장용)
    bgr_array = cv2.cvtColor(rgb_only, cv2.COLOR_RGB2BGR)
    
    return bgr_array

# ==========================================
# --- 3. 샘플 생성 메인 로직 ---
# ==========================================

def generate_sample(sample_idx):
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.81)
    plane_id = pb.loadURDF("plane.urdf")

    pb.changeVisualShape(
    plane_id, -1,
    rgbaColor=[0.85, 0.88, 0.92, 1.0]  # 밝은 회색톤
)
    folder_path = os.path.join(OUTPUT_DIR, f"GEN_{sample_idx:04d}")
    os.makedirs(folder_path, exist_ok=True)
    
    #구조 생성
    st_types = [
        'pyramid', 'tower', 'grid_tower', 'overhang', 'wall', 'spire', 'zigzag_tower' , 'leaning_grid_tower'
    ]

    # st_types = [
    #     'wall'
    # ]
    chosen = np.random.choice(st_types)
    positions = generate_structure_coords(chosen)
    meta = get_structure_metadata(positions)

    # 블록 배치
    block_ids = [
    create_block(p, PASTEL_COLORS[np.random.randint(len(PASTEL_COLORS))])
    for p in positions
]
    for _ in range(60): pb.stepSimulation() # 안정화

    actual_positions = []

    for bid in block_ids:
        pos, orn = pb.getBasePositionAndOrientation(bid)

        # 쿼터니언 → 오일러 각
        roll, pitch, yaw = pb.getEulerFromQuaternion(orn)

        actual_positions.append({
            "id": int(bid),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw)
        })
        # ---------------------------
    # 🔥 좌표 JSON 저장 (여기로 이동!)
    # ---------------------------
    

        coords_path = os.path.join(folder_path, "coords.json")

    with open(coords_path, "w") as f:
        json.dump(actual_positions, f, indent=2)
    
    # 실제 배치된 좌표를 기반으로 메타데이터 계산
    meta = get_structure_metadata([
        (p["x"], p["y"], p["z"]) for p in actual_positions
])

    # 3. 이미지 렌더링 및 저장 (영상 녹화 로직 제외)
    # Top View (위에서 수직)
    cam_x_offset = np.random.uniform(0.1, 0.2) * np.random.choice([-1, 1])
    cam_y_offset = np.random.uniform(0.1, 0.2) * np.random.choice([-1, 1])
    
    top_img = render_camera_with_ranges(
        cam_x_range=[cam_x_offset, cam_x_offset], cam_y_range=[cam_y_offset, cam_y_offset], cam_z_range=[4.0, 4.5],
        tgt_x_range=[0.0, 0.0], tgt_y_range=[0.0, 0.0], tgt_z_range=[0.1, 0.2],
        fov_range=[65, 60]
    )
    
    # Front View
    front_img = render_camera_with_ranges(
        cam_x_range=[-2.5,2.5], cam_y_range=[-2.0, -2.5], 
        cam_z_range=[2.5, 2.5],
        tgt_x_range=[-0.1, 0.1], tgt_y_range=[-0.1, 0.1], 
        tgt_z_range=[0.0, 0.6],
        fov_range=[75, 60] # 광각 화각
    )

    cv2.imwrite(os.path.join(folder_path, "top.png"), top_img)
    cv2.imwrite(os.path.join(folder_path, "front.png"), front_img)

    initial_positions = {}
    for bid in block_ids:
        pos, _ = pb.getBasePositionAndOrientation(bid)
        initial_positions[bid] = np.array(pos)
    
    # 물리 시뮬레이션 (불안정성 점수 측정)
    for _ in range(240 * 4): pb.stepSimulation()
    
    is_unstable = 0
    max_dist = 0
    for bid in block_ids:
        final_pos, _ = pb.getBasePositionAndOrientation(bid)
        dist = np.linalg.norm(np.array(final_pos) - initial_positions[bid])
        if dist > max_dist: max_dist = dist # 기록용
        
        if dist >= THRESHOLD_DIST: 
            is_unstable = 1
            break # 하나라도 무너지면 검사 중단

    # 라벨링 마무리
    meta.update({
        "sample_id": f"GEN_{sample_idx:04d}",
        "structure_type": chosen,
        "instability_score": round(float(max_dist), 4), # 가장 많이 움직인 거리 저장
        "label": "unstable" if is_unstable == 1 else "stable"
    })
    
    pb.disconnect()
    return meta

# ==========================================
# --- 4. 수정된 메타데이터 계산 (라벨링 간소화) ---
# ==========================================

def get_structure_metadata(block_positions):
    if len(block_positions) == 0:
        return {"total_blocks": 0, "max_height": 0, "l0_bounding_area": 0, "level_counts": [], "label": "stable"}
        
    pos_np = np.array(block_positions)
    
    # 1. 층별 인덱스 생성 (BLOCK_Z 기준)
    # 기존 코드의 'half', 'BLOCK_SIZE' 중복 및 미정의 오류 수정
    z_indices = np.round((pos_np[:, 2] - (BLOCK_Z / 2)) / BLOCK_Z).astype(int)
    unique_levels = np.sort(np.unique(z_indices))

    # 2. 0층 가두리 면적 계산 (Bounding Area)
    l0_mask = z_indices == 0
    l0_pos = pos_np[l0_mask]
    
    if len(l0_pos) > 0:
        # 블록의 중심좌표에서 half_extent만큼 확장하여 실제 점유 면적 계산
        min_x, max_x = np.min(l0_pos[:, 0]) - BLOCK_X/2, np.max(l0_pos[:, 0]) + BLOCK_X/2
        min_y, max_y = np.min(l0_pos[:, 1]) - BLOCK_Y/2, np.max(l0_pos[:, 1]) + BLOCK_Y/2
        area = (max_x - min_x) * (max_y - min_y)
    else: 
        area = 0.0

    # 3. 층별 블록 분포 계산
    level_counts = [int(np.sum(z_indices == lv)) for lv in unique_levels]

    return {
        "total_blocks": len(pos_np),
        "max_height": round(float(np.max(pos_np[:, 2]) + BLOCK_Z/2), 3),
        "l0_bounding_area": round(area, 4),
        "level_counts": level_counts
    }

# ==========================================
# --- 5. 수정된 실행 및 데이터 저장 ---
# ==========================================

if __name__ == "__main__":
    setup_directories()
    results = []
    
    for i in range(NUM_SAMPLES):
        print(f"Generating {i+1}/{NUM_SAMPLES}...")
        # generate_sample 내부에서 meta.update 되는 부분도 고려하여 구조 수정
        # (기존 generate_sample 코드 내 meta.update 부분을 아래 순서에 맞게 적용)
        sample_meta = generate_sample(i)
        results.append(sample_meta)
        
    df = pd.DataFrame(results)

    # 요청하신 5가지 핵심 컬럼만 정의
    ordered_columns = [
        "sample_id",            # 1. 식별자
        "structure_type",       # 2. 구조 타입
        "label"
    ]

    # 실제 df에서 해당 컬럼만 추출
    final_df = df[[col for col in ordered_columns if col in df.columns]]

    # 저장
    csv_path = os.path.join(OUTPUT_DIR, "physics_labels.csv")
    final_df.to_csv(csv_path, index=False)
    
    print(f"\n✅ 라벨링 수정 및 데이터셋 생성 완료!")
    print(f"📊 최종 컬럼: {list(final_df.columns)}")