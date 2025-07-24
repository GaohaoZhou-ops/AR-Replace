import cv2
import numpy as np
import math

# ==============================================================================
# 全局超参数 (Global Hyperparameters)
# 通过调整这些值，可以改变算法的灵敏度、稳定性和性能
# ==============================================================================

# --- 匹配与识别阈值 ---
# MIN_MATCH_COUNT: 执行单应性矩阵计算所需的最小“良好匹配”数量。
# 这是判断“是否在视频帧中找到了模板图像”的核心阈值。
#   - 增大此值: 会使识别更加“严格”和“稳定”。只有当匹配质量非常高时，程序才会认为找到了目标。
#               可以有效减少因少量错误匹配导致的误识别。但可能导致在遮挡、模糊或距离远的情况下难以识别目标。
#   - 减小此值: 会使识别更加“灵敏”。即使只有少量匹配点，程序也会尝试计算变换。
#               可以提高在困难条件下的识别率，但同时也显著增加了误识别和模型抖动的风险。
MIN_MATCH_COUNT = 10

# LOWE_RATIO_THRESHOLD: Lowe's Ratio Test的比例阈值，用于筛选匹配点。
# 一个匹配(m)只有在它的距离远小于次优匹配(n)的距离时 (m.distance < ratio * n.distance)，才被认为是“良好”的。
# 这是区分“明确匹配”和“模糊匹配”的关键。
#   - 增大此值 (如 0.8): 会放宽筛选条件，保留更多的匹配点。这在模板特征较少时可能有帮助，但会引入更多不确定或错误的匹配，可能降低最终变换的准确性。
#   - 减小此值 (如 0.65): 会收紧筛选条件，只保留非常明确、高质量的匹配点。这能有效提升单应性矩阵计算的鲁棒性，减少抖动和变形，但可能在特征不清晰时导致匹配点数量不足。
LOWE_RATIO_THRESHOLD = 0.7

# RANSAC_REPROJ_THRESHOLD: RANSAC算法的重投影误差阈值（单位：像素）。
# 在计算单应性矩阵时，RANSAC会区分“内点”（inliers）和“外点”（outliers）。一个匹配点如果其投影误差小于此阈值，则被视为“内点”。
#   - 增大此值 (如 5.0): 允许匹配点有更大的位置误差。这使得算法对噪声和轻微不准确的匹配更加宽容，但在某些情况下可能会导致计算出一个不太精确的变换矩阵。
#   - 减小此值 (如 3.0): 要求匹配点的位置非常精确。这会剔除更多有偏差的匹配点，如果“良好匹配”的质量足够高，通常能得到更精确、更稳定的变换结果。但如果匹配点本身就不够精确，过小的值可能导致无法计算出单应性矩阵。
RANSAC_REPROJ_THRESHOLD = 4.0


# --- 失真检测阈值 ---
# MIN_ANGLE_THRESHOLD: 替换图像四边形所允许的最小内角（单位：度）。
# 用于防止变换后的四边形出现过于尖锐的“挤压角”，例如变成一个漏斗形或三角形。
#   - 增大此值 (如 45.0): 角度限制更严格，只允许形状更接近“方正”的四边形通过。可以有效避免大部分严重变形，但也可能拒绝一些因为正常透视效应产生的大角度倾斜。
#   - 减小此值 (如 20.0): 角度限制更宽松，允许更大程度的透视变形。可以适应更极端的拍摄角度，但过小的值会失去防止“漏斗形”等严重失真的作用。
MIN_ANGLE_THRESHOLD = 35.0

# MAX_ANGLE_THRESHOLD: 替换图像四边形所允许的最大内角（单位：度）。
# 用于防止变换后的四边形某个角张得过开，失去四边形的基本形态。
#   - 增大此值 (如 150.0): 限制更宽松，允许更开放的角。
#   - 减小此值 (如 120.0): 限制更严格，要求四边形更加“凸”，避免出现接近平角的奇怪形状。
MAX_ANGLE_THRESHOLD = 130.0


# --- 特征提取参数 ---
# ORB_FEATURE_COUNT: ORB算法尝试检测的最大特征点数量。
# 这会影响模板图像和视频每一帧中用于匹配的“原材料”数量。
#   - 增大此值 (如 3000): ORB会尝试从图像中提取更多特征点。这对于特征丰富的大图像非常有用，可以提供更多匹配候选，从而提高识别的成功率和稳定性。缺点是会增加计算量，可能稍微降低帧率。
#   - 减小此值 (如 1000): 提取的特征点更少。可以显著提高计算速度和帧率，但对于特征较少或较小的模板，可能因为找不到足够的特征点而导致匹配失败。
ORB_FEATURE_COUNT = 2000

# ==============================================================================

orb = cv2.ORB_create(nfeatures=ORB_FEATURE_COUNT)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# 计算角度
def get_angle(p1, p2, p3):
    """计算由三点p1, p2, p3构成的角（角在p2点）"""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag_v1 == 0 or mag_v2 == 0:
        return 0
    
    arg = dot_product / (mag_v1 * mag_v2)
    arg = max(-1.0, min(1.0, arg))  # 将参数限制在 [-1.0, 1.0] 范围内，防止浮点精度错误
    
    angle_rad = math.acos(arg)
    return math.degrees(angle_rad)


def is_transformation_valid(points):
    """检查变换后的四边形是否过度扭曲"""
    # 确保是4个点
    if len(points) != 4:
        return False
        
    # 1. [可选] 检查四边形是否为凸多边形
    to_int_points = np.int32(points) # isContourConvex 需要整数点
    if not cv2.isContourConvex(to_int_points):
        return False

    # 2. 检查内角
    p = [ (pt[0][0], pt[0][1]) for pt in points]
    
    angles = []
    # 顺序: p0 -> p1 -> p2 -> p3 -> p0
    angles.append(get_angle(p[3], p[0], p[1]))
    angles.append(get_angle(p[0], p[1], p[2]))
    angles.append(get_angle(p[1], p[2], p[3]))
    angles.append(get_angle(p[2], p[3], p[0]))

    for angle in angles:
        if angle < MIN_ANGLE_THRESHOLD or angle > MAX_ANGLE_THRESHOLD:
            print(f"检测到无效角度: {angle:.2f}度")
            return False
            
    return True


def create_composite_b(template_a, template_b):
    h_a, w_a, _ = template_a.shape
    h_b, w_b, _ = template_b.shape
    composite_b = np.zeros((h_a, w_a, 3), dtype=np.uint8)
    scale = min(w_a / w_b, h_a / h_b)
    new_w, new_h = int(w_b * scale), int(h_b * scale)
    resized_b = cv2.resize(template_b, (new_w, new_h))
    x_offset, y_offset = (w_a - new_w) // 2, (h_a - new_h) // 2
    composite_b[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_b
    return composite_b


def load_templates(path_a, path_b):
    template_a = cv2.imread(path_a, cv2.IMREAD_COLOR)
    original_b = cv2.imread(path_b, cv2.IMREAD_COLOR)
    if template_a is None or original_b is None:
        print("错误: 无法加载一个或多个模板图像。")
        exit()
    composite_template_b = create_composite_b(template_a, original_b)
    kp_a, des_a = orb.detectAndCompute(template_a, None)
    if des_a is None:
        print("错误: 无法在模板A中找到足够的特征点。")
        exit()
    return template_a, composite_template_b, kp_a, des_a


def main():
    template_a, template_b_composite, kp_a, des_a = load_templates('./resources/template_1.png', './resources/replace_template.png')
    h, w, _ = template_a.shape
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头。")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        final_frame = frame.copy()

        # 仅当在当前帧中检测到特征点时才进行匹配
        if des_frame is not None and len(des_frame) > 0:
            good_matches = []
            
            # 1. 使用 k-Nearest Neighbor 匹配
            if des_a is not None:
                matches = bf.knnMatch(des_a, des_frame, k=2)
                # 2. 应用 Lowe's Ratio Test 筛选出好的匹配, 确保 matches 中的每个元素都是一个包含两个匹配的元组
                for m, n in filter(lambda x: len(x) == 2, matches):
                    if m.distance < LOWE_RATIO_THRESHOLD * n.distance:
                        good_matches.append(m)
            print(f"Good matchs count: {len(good_matches)}")

            # 3. 如果有足够多的好匹配，则计算单应性矩阵
            if len(good_matches) > MIN_MATCH_COUNT:
                # 从 good_matches 中提取匹配点的坐标
                src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # 计算单应性矩阵
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)

                if M is not None:
                    # 获取模板A的角点，并将其变换到视频帧中
                    pts_corners_a = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst_corners = cv2.perspectiveTransform(pts_corners_a, M)

                    # 4. 在替换前验证变换的有效性
                    if is_transformation_valid(dst_corners):
                        frame_h, frame_w, _ = frame.shape
                        warped_b = cv2.warpPerspective(template_b_composite, M, (frame_w, frame_h))
                        mask_template = np.ones((h, w), dtype=np.uint8) * 255
                        warped_mask = cv2.warpPerspective(mask_template, M, (frame_w, frame_h))
                        mask_inv = cv2.bitwise_not(warped_mask)
                        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                        warped_b_fg = cv2.bitwise_and(warped_b, warped_b, mask=warped_mask)
                        final_frame = cv2.add(frame_bg, warped_b_fg)
                        # 绘制绿色轮廓表示检测成功且有效
                        cv2.polylines(final_frame, [np.int32(dst_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    # 如果变换无效，则不执行任何操作，final_frame 保持为原始帧

        cv2.imshow('AR Replacement with Distortion Check', final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    main()
