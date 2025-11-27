import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


# 读取原始图像
original_image = cv2.imread(r'D:\Biancheng\Opencv\Xiaomai\7cb54c872085081fe3fe4b582166fb8c.jpg')
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# 高斯模糊 + Canny边缘检测
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
edges = cv2.Canny(blurred_image, 50, 150)

# 找到最大轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    main_contour = max(contours, key=cv2.contourArea)
else:
    print("未检测到任何轮廓")
    exit()

# 计算弯曲度
def calculate_curvature(contour):
    length = cv2.arcLength(contour, closed=False)
    start_point = contour[0][0]
    end_point = contour[-1][0]
    straight_distance = np.linalg.norm(end_point - start_point)
    return length / straight_distance

curvature = calculate_curvature(main_contour)

# 显示交互界面，标注关键点
fig, ax = plt.subplots(figsize=(10, 5))
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Click to mark keypoints (press "w" to finish one curve, "q" to finish all)')
plt.axis('off')

all_keypoints = []
current_keypoints = []

def on_click(event):
    if event.inaxes == ax:
        x, y = event.xdata, event.ydata
        current_keypoints.append((int(x), int(y)))
        ax.plot(x, y, 'ro')
        plt.draw()

def on_key(event):
    if event.key == 'w' and current_keypoints:
        all_keypoints.append(current_keypoints.copy())
        current_keypoints.clear()
        print("Curve completed.")
    elif event.key == 'q' and all_keypoints:
        plt.close()

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

# 贝塞尔曲线拟合
def bezier_curve(points, n_points=100):
    if len(points) < 2:
        return np.array(points)
    points = np.array(points)
    t = np.linspace(0, 1, n_points)
    curve = np.zeros((n_points, 2))
    for i in range(n_points):
        curve[i] = np.sum(np.array([
            np.math.comb(len(points) - 1, j) * (1 - t[i]) ** (len(points) - 1 - j) * t[i] ** j * points[j]
            for j in range(len(points))
        ]), axis=0)
    return curve.astype(int)

fitted_curves_image = original_image.copy()
results = []

for idx, keypoints in enumerate(all_keypoints):
    if len(keypoints) >= 2:
        keypoints_sorted = sorted(keypoints, key=lambda x: x[0])
        bezier_points = bezier_curve(keypoints_sorted, n_points=200)

        # 绘制曲线
        for i in range(len(bezier_points) - 1):
            cv2.line(fitted_curves_image, tuple(bezier_points[i]), tuple(bezier_points[i + 1]), (0, 255, 0), 3)

        start_point = tuple(bezier_points[0])
        end_point = tuple(bezier_points[-1])
        cv2.line(fitted_curves_image, start_point, end_point, (255, 0, 0), 2)

        # ---------- 1. 起点角度 ----------
        tangent_vector_start = np.array(bezier_points[1]) - np.array(bezier_points[0])
        vertical_vector = np.array([0, 1])
        angle_start = np.degrees(np.arccos(np.dot(tangent_vector_start, vertical_vector) /
                                           (np.linalg.norm(tangent_vector_start) * np.linalg.norm(vertical_vector))))
        if angle_start > 90:
            angle_start = 180 - angle_start
        angle_start = round(angle_start, 2)

        # ---------- 2. 中点切线角度 ----------
        mid_index = len(bezier_points) // 2
        tangent_vector_mid = np.array(bezier_points[mid_index+1]) - np.array(bezier_points[mid_index])
        angle_mid = np.degrees(np.arccos(np.dot(tangent_vector_mid, vertical_vector) /
                                         (np.linalg.norm(tangent_vector_mid) * np.linalg.norm(vertical_vector))))
        if angle_mid > 90:
            angle_mid = 180 - angle_mid
        angle_mid = round(angle_mid, 2)

        # ---------- 3. 整体趋势角度（线性回归法） ----------
        # 用 numpy 的 polyfit 拟合直线 (y = kx + b)
        X = bezier_points[:, 0]
        Y = bezier_points[:, 1]
        slope, intercept = np.polyfit(X, Y, 1)

        # 由斜率计算角度
        angle_trend = np.degrees(np.arctan(slope))
        if angle_trend < 0:
            angle_trend += 180
        if angle_trend > 90:
            angle_trend = 180 - angle_trend
        angle_trend = round(angle_trend, 2)

        if angle_trend < 0:
            angle_trend += 180
        if angle_trend > 90:
            angle_trend = 180 - angle_trend
        angle_trend = round(angle_trend, 2)

        # 绘制角度文字
        img_pil = Image.fromarray(cv2.cvtColor(fitted_curves_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.load_default()
        draw.text((start_point[0], start_point[1] - 10),
                  f'Start: {angle_start}°', font=font, fill=(255, 0, 0))
        draw.text((start_point[0], start_point[1] - 30),
                  f'Mid: {angle_mid}°', font=font, fill=(0, 0, 255))
        draw.text((start_point[0], start_point[1] - 50),
                  f'Trend: {angle_trend}°', font=font, fill=(0, 128, 0))
        fitted_curves_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 保存结果
        results.append((idx+1, angle_start, angle_mid, angle_trend))

# 显示结果图
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(fitted_curves_image, cv2.COLOR_BGR2RGB))
plt.title('Bezier Curves with Angles')
plt.axis('off')
plt.show()

# 保存结果
with open('Out/curvature_result.txt', 'w') as f:
    f.write(f"曲线弯曲度: {curvature}\n")
    for idx, a1, a2, a3 in results:
        f.write(f"第 {idx} 条曲线：起点角度={a1}°, 中点角度={a2}°, 整体趋势角度={a3}°\n")
