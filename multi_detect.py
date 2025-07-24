import cv2
import numpy as np
import math

# ==============================================================================
# 全局设置 (Global Settings)
# ==============================================================================

# --- 默认超参数 ---
# 如果某个模板对没有在配置中指定自己的参数，将使用这些默认值。
DEFAULT_PARAMS = {
    "MIN_MATCH_COUNT": 10,
    "LOWE_RATIO_THRESHOLD": 0.7,
    "RANSAC_REPROJ_THRESHOLD": 4.0,
    "MIN_ANGLE_THRESHOLD": 35.0,
    "MAX_ANGLE_THRESHOLD": 130.0
}

# --- 特征提取器 ---
# ORB的特征点数量仍然是全局的，因为它在程序开始时初始化一次，
# 并用于处理视频的每一帧。为每个模板使用不同的ORB实例会显著增加计算开销。
# 建议设置一个较高的通用值，然后通过其他参数微调每个模板的匹配。
ORB_FEATURE_COUNT = 2000

# 初始化ORB特征检测器和BFMatcher
orb = cv2.ORB_create(nfeatures=ORB_FEATURE_COUNT)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


# ==============================================================================
# 类与函数定义 (Class and Function Definitions)
# ==============================================================================

class Template:
    """
    一个用于存储模板对所有相关信息的类。
    每个实例现在也包含自己的一套匹配和验证参数。
    """
    def __init__(self, name, pair_config):
        self.name = name
        self.template_img_path = pair_config['template_path']
        self.replacement_img_path = pair_config['replace_path']

        # --- 参数加载 ---
        # 获取用户为这个模板定义的参数，如果未定义则使用空字典
        user_params = pair_config.get("params", {})
        # 使用用户定义的值，如果缺失则从DEFAULT_PARAMS中获取
        self.min_match_count = user_params.get("MIN_MATCH_COUNT", DEFAULT_PARAMS["MIN_MATCH_COUNT"])
        self.lowe_ratio_threshold = user_params.get("LOWE_RATIO_THRESHOLD", DEFAULT_PARAMS["LOWE_RATIO_THRESHOLD"])
        self.ransac_reproj_threshold = user_params.get("RANSAC_REPROJ_THRESHOLD", DEFAULT_PARAMS["RANSAC_REPROJ_THRESHOLD"])
        self.min_angle_threshold = user_params.get("MIN_ANGLE_THRESHOLD", DEFAULT_PARAMS["MIN_ANGLE_THRESHOLD"])
        self.max_angle_threshold = user_params.get("MAX_ANGLE_THRESHOLD", DEFAULT_PARAMS["MAX_ANGLE_THRESHOLD"])

        # --- 图像加载与处理 ---
        self.template_img = cv2.imread(self.template_img_path, cv2.IMREAD_COLOR)
        self.original_replacement_img = cv2.imread(self.replacement_img_path, cv2.IMREAD_COLOR)
        if self.template_img is None or self.original_replacement_img is None:
            raise IOError(f"错误: 无法加载模板 '{name}' 的图像路径: {self.template_img_path} 或 {self.replacement_img_path}")

        self.h, self.w, _ = self.template_img.shape
        self.sized_replacement_img = self._create_sized_replacement()
        self.keypoints, self.descriptors = orb.detectAndCompute(self.template_img, None)
        if self.descriptors is None:
            raise ValueError(f"错误: 无法在模板 '{name}' ({self.template_img_path}) 中找到足够的特征点。")

    def _create_sized_replacement(self):
        h_a, w_a, _ = self.template_img.shape
        h_b, w_b, _ = self.original_replacement_img.shape
        composite_b = np.zeros((h_a, w_a, 3), dtype=np.uint8)
        scale = min(w_a / w_b, h_a / h_b)
        new_w, new_h = int(w_b * scale), int(h_b * scale)
        resized_b = cv2.resize(self.original_replacement_img, (new_w, new_h))
        x_offset, y_offset = (w_a - new_w) // 2, (h_a - new_h) // 2
        composite_b[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_b
        return composite_b


def get_angle(p1, p2, p3):
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag_v1 == 0 or mag_v2 == 0:
        return 0
    arg = max(-1.0, min(1.0, dot_product / (mag_v1 * mag_v2)))
    return math.degrees(math.acos(arg))


def is_transformation_valid(points, min_angle, max_angle):
    """
    检查变换后的四边形是否过度扭曲。
    现在接收角度阈值作为参数。
    """
    if len(points) != 4:
        return False
    if not cv2.isContourConvex(np.int32(points)):
        return False
    p = [(pt[0][0], pt[0][1]) for pt in points]
    angles = [
        get_angle(p[3], p[0], p[1]),
        get_angle(p[0], p[1], p[2]),
        get_angle(p[1], p[2], p[3]),
        get_angle(p[2], p[3], p[0])
    ]
    for angle in angles:
        if not (min_angle < angle < max_angle):
            # print(f"检测到无效角度: {angle:.2f}度")
            return False
    return True


def setup_templates(template_pairs_config):
    templates = []
    print("正在加载和预处理模板...")
    for name, pair_config in template_pairs_config.items():
        try:
            template = Template(name, pair_config)
            templates.append(template)
            print(f"- 成功加载模板 '{name}' (匹配数 > {template.min_match_count}, 比例 < {template.lowe_ratio_threshold})")
        except (IOError, ValueError) as e:
            print(e)
    print("模板加载完成。")
    return templates


def main():
    """
    主函数，执行视频捕获和AR替换。
    """
    # ==========================================================================
    # --- 模板配置中心 ---
    # 在这里添加或修改您想要识别和替换的图像对。
    # 新增 "params" 键，您可以在其中为特定模板覆盖默认参数。
    # ==========================================================================
    TEMPLATE_PAIRS = {
        "book_cover": {
            "template_path": "./resources/template_1.png",
            "replace_path": "./resources/replace_template_1.png",
            # 此模板未使用自定义参数，将全部采用 DEFAULT_PARAMS 的值
        },
        "card": {
            "template_path": "./resources/template_2.png",
            "replace_path": "./resources/replace_template_2.png",
            "params": {
                # 这个模板比较小，特征少，我们放宽一些条件
                "MIN_MATCH_COUNT": 10,          # 降低最小匹配数要求
                "LOWE_RATIO_THRESHOLD": 0.70,   # 允许质量稍差一些的匹配
                "MIN_ANGLE_THRESHOLD": 25.0,    # 允许更大的透视形变
                "MAX_ANGLE_THRESHOLD": 130,
            }
        },
    }

    templates = setup_templates(TEMPLATE_PAIRS)
    if not templates:
        print("错误: 没有任何模板被成功加载。程序退出。")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头。")
        return

    # --- 主循环 ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        final_frame = frame.copy()

        if des_frame is not None and len(des_frame) > 0:
            for tpl in templates:
                good_matches = []
                matches = bf.knnMatch(tpl.descriptors, des_frame, k=2)
                
                # 使用模板自己的Lowe's Ratio阈值
                for m, n in filter(lambda x: len(x) == 2, matches):
                    if m.distance < tpl.lowe_ratio_threshold * n.distance:
                        good_matches.append(m)
                
                # 使用模板自己的最小匹配数阈值
                if len(good_matches) > tpl.min_match_count:
                    src_pts = np.float32([tpl.keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # 使用模板自己的RANSAC重投影误差阈值
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, tpl.ransac_reproj_threshold)

                    if M is not None:
                        pts_corners_tpl = np.float32([[0, 0], [0, tpl.h - 1], [tpl.w - 1, tpl.h - 1], [tpl.w - 1, 0]]).reshape(-1, 1, 2)
                        dst_corners = cv2.perspectiveTransform(pts_corners_tpl, M)

                        # 使用模板自己的角度阈值进行验证
                        if is_transformation_valid(dst_corners, tpl.min_angle_threshold, tpl.max_angle_threshold):
                            frame_h, frame_w, _ = final_frame.shape
                            warped_replacement = cv2.warpPerspective(tpl.sized_replacement_img, M, (frame_w, frame_h))
                            mask_template = np.ones((tpl.h, tpl.w), dtype=np.uint8) * 255
                            warped_mask = cv2.warpPerspective(mask_template, M, (frame_w, frame_h))
                            mask_inv = cv2.bitwise_not(warped_mask)
                            frame_bg = cv2.bitwise_and(final_frame, final_frame, mask=mask_inv)
                            replacement_fg = cv2.bitwise_and(warped_replacement, warped_replacement, mask=warped_mask)
                            final_frame = cv2.add(frame_bg, replacement_fg)
                            cv2.polylines(final_frame, [np.int32(dst_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Multi-Template AR with Custom Params', final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()