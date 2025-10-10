import sys
import cv2
import numpy as np
import torch
import os
import time
from collections import defaultdict, deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QTextBrowser
from PyQt5.QtGui import QImage, QPixmap, QTextDocument
from PyQt5.QtCore import QTimer, Qt, QUrl
import lib.summertts as summertts

# 添加当前目录到Python路径
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lfd.execution.utils import load_checkpoint
from lfd.data_pipeline.augmentation import *
from ui import Ui_MainWindow

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载原始人脸检测模型
from WIDERFACE_LFD_XS import config_dict, prepare_model

prepare_model()
param_file_path = 'facemodel/epoch_1000.pth'
load_checkpoint(config_dict['model'], load_path=param_file_path, strict=True)
config_dict['model'] = config_dict['model'].to(device)

# 创建输出目录
os.makedirs("face_output", exist_ok=True)
os.makedirs("monitor", exist_ok=True)

class KalmanTracker:
    """卡尔曼滤波器跟踪器"""

    def __init__(self, id):
        self.id = id  # 跟踪器唯一标识符
        # 初始化卡尔曼滤波器
        self.kalman = cv2.KalmanFilter(6, 2)  # 6个状态变量(x,y,vx,vy,ax,ay)，2个测量变量(x,y)

        # 状态转移矩阵 (假设匀速运动模型)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]], dtype=np.float32)

        # 测量矩阵
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]], dtype=np.float32)

        # 过程噪声协方差
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1

        # 测量噪声协方差
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5

        # 后验误差协方差
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32)

        # 初始状态
        self.kalman.statePost = np.zeros((6, 1), dtype=np.float32)

        # 跟踪状态
        self.tracked = False
        self.missed_frames = 0
        self.max_missed_frames = 10  # 最多允许丢失的帧数
        self.track_history = deque(maxlen=20)  # 存储跟踪历史用于轨迹绘制
        self.color = np.random.randint(0, 255, 3).tolist()  # 为每个跟踪器分配随机颜色
        self.bbox = None  # 存储当前边界框
        self.updated_this_frame = False  # 标记当前帧是否被更新
        
        # 增强型参数
        self.adaptive_noise = True  # 启用自适应噪声
        self.base_process_noise = 0.1  # 基础过程噪声
        self.velocity_threshold = 5.0  # 速度阈值

    def update(self, measurement, bbox=None):
        """更新跟踪器"""
        self.updated_this_frame = False  # 重置更新标志
        
        if measurement is not None and bbox is not None:
            # 有检测结果时进行更新
            measurement = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
            self.bbox = bbox  # 存储边界框
            
            # 自适应过程噪声
            if self.adaptive_noise:
                _, _, vx, vy = self.get_state()
                velocity = np.sqrt(vx**2 + vy**2)
                noise_scale = max(1.0, velocity / self.velocity_threshold)
                self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * (self.base_process_noise * noise_scale)
                
            self.kalman.correct(measurement)
            self.tracked = True
            self.missed_frames = 0
            self.updated_this_frame = True  # 标记为已更新
        else:
            # 没有检测结果时只进行预测
            self.missed_frames += 1
            if self.missed_frames > self.max_missed_frames:
                self.tracked = False

        # 预测下一状态
        prediction = self.kalman.predict()
        self.track_history.append((prediction[0], prediction[1]))
        return prediction

    def get_state(self):
        """获取当前状态"""
        state = self.kalman.statePost
        return (state[0][0], state[1][0], state[2][0], state[3][0])
    
    def get_predicted_bbox(self):
        """获取基于预测的边界框"""
        if self.bbox is None:
            return None
            
        # 使用预测的中心点，但保持原始宽高
        x, y, _, _ = self.get_state()
        w, h = self.bbox[2], self.bbox[3]
        return (x - w/2, y - h/2, w, h)


class MultiFaceTracker:
    """多人脸跟踪器管理器"""

    def __init__(self):
        self.trackers = {}  # 存储所有活动的跟踪器 {id: tracker}
        self.next_id = 0  # 下一个可用的跟踪器ID

    def update(self, detections):
        """更新所有跟踪器"""
        active_ids = set()

        # 如果没有检测到人脸，只更新现有跟踪器
        if not detections:
            for tracker_id, tracker in list(self.trackers.items()):
                tracker.update(None)
                if not tracker.tracked:
                    del self.trackers[tracker_id]
            return self.trackers

        # 将检测结果与现有跟踪器匹配
        matched_pairs = self._match_detections_to_trackers(detections)

        # 更新匹配的跟踪器
        for detection_idx, tracker_id in matched_pairs:
            center = self._get_detection_center(detections[detection_idx])
            bbox = detections[detection_idx][2:6]  # 提取边界框信息
            self.trackers[tracker_id].update(center, bbox)
            active_ids.add(tracker_id)

        # 为未匹配的检测创建新跟踪器
        for i, det in enumerate(detections):
            if i not in [pair[0] for pair in matched_pairs]:
                new_tracker = KalmanTracker(self.next_id)
                center = self._get_detection_center(det)
                bbox = det[2:6]  # 提取边界框信息
                new_tracker.update(center, bbox)
                self.trackers[self.next_id] = new_tracker
                active_ids.add(self.next_id)
                self.next_id += 1

        # 移除长时间未更新的跟踪器
        for tracker_id, tracker in list(self.trackers.items()):
            if tracker_id not in active_ids:
                tracker.update(None)
                if not tracker.tracked:
                    del self.trackers[tracker_id]

        return self.trackers

    def _get_detection_center(self, detection):
        """获取检测框的中心点"""
        return (detection[2] + detection[4] / 2, detection[3] + detection[5] / 2)

    def _match_detections_to_trackers(self, detections):
        """使用匈牙利算法匹配检测和跟踪器"""
        if not self.trackers:
            return []

        # 计算所有检测与跟踪器预测位置之间的IOU
        cost_matrix = []
        for det in detections:
            det_bbox = (det[2], det[3], det[4], det[5])
            row = []
            for tracker in self.trackers.values():
                # 获取跟踪器预测的边界框
                pred_bbox = tracker.get_predicted_bbox()
                
                if pred_bbox is None:
                    # 如果没有预测框，使用大距离值
                    row.append(1000.0)
                    continue
                
                # 计算IOU
                iou = self._calculate_iou(det_bbox, pred_bbox)
                # 使用1-IOU作为代价（越小越好）
                cost = 1.0 - iou
                row.append(cost)
            cost_matrix.append(row)

        # 如果没有跟踪器，返回空列表
        if not cost_matrix or not cost_matrix[0]:
            return []

        # 使用匈牙利算法进行匹配
        try:
            from scipy.optimize import linear_sum_assignment
            det_indices, tracker_indices = linear_sum_assignment(cost_matrix)
        except ImportError:
            det_indices, tracker_indices = self._simple_greedy_matching(cost_matrix)

        # 只保留IOU大于阈值的匹配
        matched_pairs = []
        for det_idx, tracker_idx in zip(det_indices, tracker_indices):
            if cost_matrix[det_idx][tracker_idx] < 0.7:  # 对应IOU>0.3
                tracker_id = list(self.trackers.keys())[tracker_idx]
                matched_pairs.append((det_idx, tracker_id))

        return matched_pairs

    def _calculate_iou(self, box1, box2):
        """计算两个边界框的IOU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算交集区域
        xx1 = max(x1, x2)
        yy1 = max(y1, y2)
        xx2 = min(x1 + w1, x2 + w2)
        yy2 = min(y1 + h1, y2 + h2)
        
        # 计算交集面积
        inter_w = max(0, xx2 - xx1)
        inter_h = max(0, yy2 - yy1)
        intersection = inter_w * inter_h
        
        # 计算并集面积
        union = w1 * h1 + w2 * h2 - intersection
        
        # 避免除以零
        if union == 0:
            return 0.0
            
        return intersection / union

    def _simple_greedy_matching(self, cost_matrix):
        """简单贪婪匹配算法"""
        matches = []
        rows = len(cost_matrix)
        cols = len(cost_matrix[0]) if rows > 0 else 0
        
        # 创建标记数组
        row_used = [False] * rows
        col_used = [False] * cols
        
        # 按代价排序所有可能的匹配
        all_matches = []
        for i in range(rows):
            for j in range(cols):
                all_matches.append((cost_matrix[i][j], i, j))
                
        # 按代价升序排序
        all_matches.sort(key=lambda x: x[0])
        
        # 选择最佳匹配
        for cost, i, j in all_matches:
            if not row_used[i] and not col_used[j] and cost < 0.7:  # IOU阈值
                matches.append((i, j))
                row_used[i] = True
                col_used[j] = True
                
        # 提取匹配索引
        if matches:
            return zip(*matches)
        return [], []


def process_frame(frame, tracker_manager, detections=None, draw_detections=False):
    """处理单个视频帧"""
    # 更新所有跟踪器
    active_trackers = tracker_manager.update(detections if detections is not None else [])

    # 在检测帧绘制检测框
    if draw_detections and detections:
        for det in detections:
            cv2.rectangle(frame,
                          (int(det[2]), int(det[3])),
                          (int(det[2] + det[4]), int(det[3] + det[5])),
                          (0, 255, 0), 2)  # 绿色检测框

    # 绘制跟踪结果
    for tracker_id, tracker in active_trackers.items():
        if tracker.tracked and tracker.bbox is not None:
            # 在非检测帧，或者检测帧中已更新的跟踪器才绘制
            if not draw_detections or (draw_detections and tracker.updated_this_frame):
                # 获取跟踪状态
                x, y, vx, vy = tracker.get_state()

                # 绘制跟踪ID和中心点
                cv2.putText(frame, f"ID:{tracker_id}", (int(x) - 20, int(y) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.circle(frame, (int(x), int(y)), 5, (0,0,255), -1)

                # 绘制速度向量（减少绘制频率）
                if len(tracker.track_history) % 3 == 0:  # 每3帧绘制一次
                    cv2.arrowedLine(frame, (int(x), int(y)),
                                    (int(x + vx * 5), int(y + vy * 5)),
                                    (0,0,255), 2)

                # 绘制跟踪轨迹（减少点数量）
                if len(tracker.track_history) > 1:
                    # 每3个点取一个点绘制
                    for i in range(1, len(tracker.track_history), 3):
                        prev = tracker.track_history[i - 1]
                        curr = tracker.track_history[i]
                        cv2.line(frame, (int(prev[0]), int(prev[1])),
                                 (int(curr[0]), int(curr[1])),
                                 (0,0,255), 1)

                # 绘制边界框
                x1, y1, w, h = tracker.bbox
                cv2.rectangle(frame,
                              (int(x1), int(y1)),
                              (int(x1 + w), int(y1 + h)),
                              (0,0,255), 2)  # 跟踪器颜色

    return frame


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 初始化变量
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.tracker_manager = MultiFaceTracker()
        self.recognizing = False
        self.recording = False
        self.monitoring = False
        self.monitor_recording = False
        self.last_face_detected_time = None
        self.frame_count = 0
        self.detection_interval = 10
        self.detections = None
        self.draw_detections = False
        self.video_writer = None
        self.monitor_video_writer = None
        self.face_counter = 0
        self.monitor_counter = 0
        self.tts_initialized = False
        self.current_tts_language = None  # 'zh' or 'en'
        
        # 连接按钮信号
        self.ui.pushButton.clicked.connect(self.start_recognition)  # 开始识别
        self.ui.pushButton_3.clicked.connect(self.stop_recognition)  # 停止识别
        self.ui.pushButton_4.clicked.connect(self.start_recording)  # 开始录像
        self.ui.pushButton_5.clicked.connect(self.save_recording)  # 保存录像
        self.ui.pushButton_2.clicked.connect(self.save_faces)  # 人脸保存
        self.ui.pushButton_7.clicked.connect(self.play_tts)  # 播放语音
        self.ui.pushButton_8.clicked.connect(self.toggle_monitor_mode)  # 监测模式
        self.ui.pushButton_6.clicked.connect(self.exit_app)  # 退出
        
        # 初始化摄像头
        self.init_camera()
        
        # 初始化TTS
        self.init_tts()
        
        # 设置初始状态
        self.ui.textBrowser_3.setText("当前为普通模式")
        
        # 开始定时器
        self.timer.start(30)  # 30ms刷新率，约33fps

    def init_camera(self):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头")
            return False
            
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        return True

    def init_tts(self):
        """初始化TTS引擎"""
        # 不立即初始化，只在需要时初始化
        pass

    def initialize_tts_engine(self, language):
        """初始化TTS引擎"""
        if self.tts_initialized and self.current_tts_language == language:
            return  # 已经初始化且语言相同，无需重新初始化
            
        # 如果已初始化但语言不同，先释放
        if self.tts_initialized:
            summertts.release_tts()
            self.tts_initialized = False
            
        # 根据语言选择模型
        if language == 'zh':
            model_path = "ttsmodel/single_speaker_fast.bin"
        else:  # English
            model_path = "ttsmodel/single_speaker_english_fast.bin"
            
        # 初始化TTS引擎
        summertts.init_tts(model_path)
        self.tts_initialized = True
        self.current_tts_language = language

    def play_tts(self):
        """播放TTS"""
        # 获取文本
        text = self.ui.textBrowser_2.toPlainText().strip()
        if not text:
            return
            
        # 检测语言
        # 简单的语言检测：如果包含中文字符则认为是中文，否则是英文
        is_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        language = 'zh' if is_chinese else 'en'
        
        # 初始化TTS引擎
        self.initialize_tts_engine(language)
        
        # 播放语音
        summertts.play_voice(text, 1.0)

    def update_frame(self):
        """更新帧"""
        if self.cap is None or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # 处理帧
        if self.recognizing:
            self.frame_count += 1
            
            # 每detection_interval帧执行一次人脸检测
            if self.frame_count % self.detection_interval == 0:
                # 执行人脸检测
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.detections = config_dict['model'].predict_for_single_image(
                    image,
                    aug_pipeline=simple_widerface_val_pipeline,
                    classification_threshold=0.5,
                    nms_threshold=0.3
                )
                self.draw_detections = True  # 标记为检测帧
                
                # 更新最后检测到人脸的时间
                if len(self.detections) > 0:
                    self.last_face_detected_time = time.time()
                    
                    # 监测模式下开始录像
                    if self.monitoring and not self.monitor_recording:
                        self.start_monitor_recording()
                        
            else:
                self.draw_detections = False
                
            # 处理帧（传递检测结果）
            processed_frame = process_frame(frame, self.tracker_manager, self.detections, self.draw_detections)
            
            # 如果在录像，写入帧
            if self.recording and self.video_writer is not None:
                self.video_writer.write(processed_frame)
                
            # 监测模式下检查是否需要停止录像
            if self.monitoring and self.monitor_recording and self.last_face_detected_time is not None:
                self.monitor_video_writer.write(processed_frame)
                if time.time() - self.last_face_detected_time > 5:  # 5秒内未检测到人脸
                    self.stop_monitor_recording()
                    
            # 显示处理后的帧
            self.display_image(processed_frame)
        else:
            # 显示原始帧
            self.display_image(frame)
            
        # 监测模式下检查是否需要开始录像
        if self.monitoring and not self.recognizing:
            self.start_recognition()

    def display_image(self, img):
        """在textBrowser中显示图像"""
        # 将OpenCV图像转换为QImage
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # 缩放图像以适应textBrowser
        scaled_pixmap = pixmap.scaled(self.ui.textBrowser.size(), Qt.KeepAspectRatio)
        self.ui.textBrowser.clear()
        self.ui.textBrowser.append('')
        self.ui.textBrowser.document().addResource(QTextDocument.ImageResource, QUrl("image"), scaled_pixmap)
        self.ui.textBrowser.setHtml('<img src="image" />')

    def start_recognition(self):
        """开始识别"""
        if self.monitoring:
            # 在监测模式下，其他按钮不可用
            if self.sender() != self.ui.pushButton_8 and self.sender() != self.ui.pushButton_6:
                self.ui.textBrowser_4.setText("当前为监测模式，需先退出监测模式")
                return
                
        self.recognizing = True
        self.ui.textBrowser_4.setText("开始识别")

    def stop_recognition(self):
        """停止识别"""
        if self.monitoring:
            # 在监测模式下，其他按钮不可用
            if self.sender() != self.ui.pushButton_8 and self.sender() != self.ui.pushButton_6:
                self.ui.textBrowser_4.setText("当前为监测模式，需先退出监测模式")
                return
                
        self.recognizing = False
        self.ui.textBrowser_4.setText("停止识别")

    def start_recording(self):
        """开始录像"""
        if self.monitoring:
            # 在监测模式下，其他按钮不可用
            if self.sender() != self.ui.pushButton_8 and self.sender() != self.ui.pushButton_6:
                self.ui.textBrowser_4.setText("当前为监测模式，需先退出监测模式")
                return
                
        if not self.recording:
            # 创建临时视频文件
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 根据识别状态设置帧率
            fps = 11.0 if self.recognizing else 30.0
            self.video_writer = cv2.VideoWriter('temp_video.mp4', fourcc, fps, (640, 480))
            self.recording = True
            self.ui.textBrowser_4.setText("开始录像")

    def save_recording(self):
        """保存录像"""
        if self.monitoring:
            # 在监测模式下，其他按钮不可用
            if self.sender() != self.ui.pushButton_8 and self.sender() != self.ui.pushButton_6:
                self.ui.textBrowser_4.setText("当前为监测模式，需先退出监测模式")
                return
                
        if self.recording and self.video_writer is not None:
            # 停止录像
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            
            # 弹出文件保存对话框
            file_path, _ = QFileDialog.getSaveFileName(self, "保存录像", "", "MP4 Files (*.mp4)")
            if file_path:
                # 移动临时文件到指定位置
                try:
                    os.rename('temp_video.mp4', file_path)
                    self.ui.textBrowser_4.setText("录像已保存")
                except Exception as e:
                    self.ui.textBrowser_4.setText(f"保存失败: {str(e)}")
            else:
                # 用户取消保存，删除临时文件
                if os.path.exists('temp_video.mp4'):
                    os.remove('temp_video.mp4')
                self.ui.textBrowser_4.setText("录像保存已取消")

    def save_faces(self):
        """保存人脸"""
        if self.monitoring:
            # 在监测模式下，其他按钮不可用
            if self.sender() != self.ui.pushButton_8 and self.sender() != self.ui.pushButton_6:
                self.ui.textBrowser_4.setText("当前为监测模式，需先退出监测模式")
                return
                
        if not self.recognizing:
            self.ui.textBrowser_4.setText("不在识别状态")
            return
            
        # 确保有检测结果
        if self.detections is None or len(self.detections) == 0:
            self.ui.textBrowser_4.setText("未检测到人脸")
            return
            
        # 保存每张人脸
        ret, frame = self.cap.read()
        if not ret:
            self.ui.textBrowser_4.setText("无法获取当前帧")
            return
            
        for i, det in enumerate(self.detections):
            x, y, w, h = int(det[2]), int(det[3]), int(det[4]), int(det[5])
            # 确保坐标在图像范围内
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            # 裁剪人脸区域
            face_img = frame[y:y+h, x:x+w]
            
            # 保存图像
            filename = f"face_output/face_{self.face_counter}.jpg"
            self.face_counter += 1
            cv2.imwrite(filename, face_img)
            
        self.ui.textBrowser_4.setText(f"已保存{len(self.detections)}张人脸图像")

    def toggle_monitor_mode(self):
        """切换监测模式"""
        self.monitoring = not self.monitoring
        if self.monitoring:
            self.ui.textBrowser_3.setText("当前为监测模式")
            # 禁用除退出外的所有按钮
            self.ui.pushButton.setEnabled(False)
            self.ui.pushButton_2.setEnabled(False)
            self.ui.pushButton_3.setEnabled(False)
            self.ui.pushButton_4.setEnabled(False)
            self.ui.pushButton_5.setEnabled(False)
            self.ui.pushButton_7.setEnabled(False)
            self.ui.pushButton_8.setText("监测模式关闭")
            
            # 自动开始识别
            self.start_recognition()
        else:
            self.ui.textBrowser_3.setText("当前为普通模式")
            # 启用所有按钮
            self.ui.pushButton.setEnabled(True)
            self.ui.pushButton_2.setEnabled(True)
            self.ui.pushButton_3.setEnabled(True)
            self.ui.pushButton_4.setEnabled(True)
            self.ui.pushButton_5.setEnabled(True)
            self.ui.pushButton_7.setEnabled(True)
            self.ui.pushButton_8.setText("监测模式开启")
            
            # 停止监测录像
            if self.monitor_recording:
                self.stop_monitor_recording()

    def start_monitor_recording(self):
        """开始监测录像"""
        if not self.monitor_recording:
            # 创建监测视频文件
            timestamp = int(time.time())
            filename = f"monitor/moni_{self.monitor_counter}.mp4"
            self.monitor_counter += 1
            
            # 使用更通用的编码格式
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 确保目录存在
            #os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.monitor_video_writer = cv2.VideoWriter(filename, fourcc, 11.0, (640, 480))
            
            # 检查VideoWriter是否成功初始化
            if not self.monitor_video_writer.isOpened():
                self.ui.textBrowser_4.setText("无法初始化视频录制")
                self.monitor_recording = False
                return
            self.monitor_recording = True
            self.ui.textBrowser_4.setText("检测到人脸并开始录制")
            self.last_face_detected_time = time.time()  # 重置时间

    def stop_monitor_recording(self):
        """停止监测录像"""
        if self.monitor_recording and self.monitor_video_writer is not None:
            self.monitor_video_writer.release()
            self.monitor_video_writer = None
            self.monitor_recording = False
            self.ui.textBrowser_4.setText("成功保存视频")

    def exit_app(self):
        """退出应用"""
        # 释放资源
        if self.cap is not None:
            self.cap.release()
            
        if self.video_writer is not None:
            self.video_writer.release()
            
        if self.monitor_video_writer is not None:
            self.monitor_video_writer.release()
            
        # 释放TTS资源
        if self.tts_initialized:
            summertts.release_tts()
            
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        # 退出应用
        QApplication.quit()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
