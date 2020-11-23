import cv2
import imutils


def show_video(video_path="./model_data/2.mp4"):
    video_capture = cv2.VideoCapture(video_path)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    size = (video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("fps: {}\nsize: {}".format(fps, size))

    total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("[INFO] {} total frames in video".format(total))

    # 设定从视频的第几帧开始读取
    frameToStart = 0
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)

    # 显示视频
    current_frame = frameToStart
    while True:
        success, frame = video_capture.read()
        print(success)
        if success == False:
            break
        # 自定义图像大小
        # h, w = frame.shape[:2]  # 三通道
        # size = (int(w * 0.5), int(h * 0.5))
        # frame = cv2.resize(frame, size)
        frame = imutils.resize(frame, width=720)
        # --------键盘控制视频---------------
        # 读取键盘值
        key = cv2.waitKey(1) & 0xff
        # 设置空格按下时暂停
        if key == ord(" "):
            cv2.waitKey(0)
        # 设置Q按下时退出
        if key == ord("q"):
            break

        # 显示当前视频已播放时间和总时间
        # 计算当前
        now_seconds = int(current_frame / fps % 60)
        now_minutes = int(current_frame / fps / 60)
        total_second = int(total / fps % 60)
        total_minutes = int(total / fps / 60)
        #   { <参数序号> : <填充> <对齐）> <宽度> <,> <.精度> <类型>}.
        Time_now_vs_total = "Time:{:>3}:{:>02} |{:>3}:{:0>2}".format(now_minutes, now_seconds, total_minutes,
                                                                     total_second)
        print(Time_now_vs_total)

        #  putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None):
        cv2.putText(frame, Time_now_vs_total, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow("frame", frame)

        # 人工对视频帧数进行计数
        current_frame += 1
        # TODO: 播放到最后，静止画面
        if current_frame == total:
            cv2.waitKey(0)
    video_capture.release()
