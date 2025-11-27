#无损截图


import cv2
import os
import csv
from datetime import datetime


def create_unique_output_folder(base_folder, prefix="frames"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{prefix}_{timestamp}"
    output_path = os.path.join(base_folder, folder_name)
    os.makedirs(output_path, exist_ok=True)
    return output_path


def capture_video_frames(video_path, base_output_folder, interval=1, image_format="png", show_preview=False):
    assert image_format.lower() in ["jpg", "png", "bmp", "tiff"], "仅支持 jpg/png/bmp/tiff 格式"

    # 创建输出目录
    output_folder = create_unique_output_folder(base_output_folder)
    csv_path = os.path.join(output_folder, "frame_info.csv")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"视频信息: {total_frames} 帧 | {fps:.2f} FPS | 分辨率: {width}x{height}")
        print(f"开始每 {interval} 帧保存一帧到目录: {output_folder}")

        frame_count = 0
        saved_count = 0
        success = True

        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["frame_number", "timestamp_sec", "file_name"])

            while success:
                success, frame = cap.read()
                if not success:
                    break

                if frame_count % interval == 0:
                    # 时间戳（单位：秒）
                    timestamp = frame_count / fps
                    frame_file = os.path.join(output_folder, f"frame_{frame_count:06d}.{image_format}")

                    # 保存图像，确保无损（特别是对 PNG）
                    if image_format.lower() == "jpg":
                        cv2.imwrite(frame_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])  # 最高质量
                    elif image_format.lower() == "png":
                        cv2.imwrite(frame_file, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 无压缩
                    else:
                        cv2.imwrite(frame_file, frame)  # BMP/TIFF默认无损

                    writer.writerow([frame_count, round(timestamp, 3), os.path.basename(frame_file)])
                    saved_count += 1

                if show_preview:
                    cv2.imshow("Frame Preview", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户终止预览。")
                        break

                if frame_count % 100 == 0:
                    print(f"已处理: {frame_count}/{total_frames} 帧 | 已保存: {saved_count}")

                frame_count += 1

        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

        print(f"\n完成! 共保存 {saved_count} 帧到目录: {output_folder}")
        print(f"帧信息已保存到 CSV 文件: {csv_path}")

    except Exception as e:
        print("发生错误:", e)


# ========== 用户配置 ==========
if __name__ == "__main__":
    video_path = r"D:\Data\Action 2025.6.5\Top\6m\Top6m-1.mp4"
    base_output_folder = r"D:\Data\Action 2025.6.5\Top\6m"
    interval = 2                  # 每隔多少帧保存一张
    image_format = "png"          # 建议用 png（无损），可选 jpg/bmp/tiff
    show_preview = False          # 是否显示帧预览

    capture_video_frames(
        video_path=video_path,
        base_output_folder=base_output_folder,
        interval=interval,
        image_format=image_format,
        show_preview=show_preview
    )
