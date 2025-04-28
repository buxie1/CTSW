import os
import re
import tempfile
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import odl
# 从 qt 模块下导入各个UI文件（由 Qt Designer 生成的类），这些类用于构造各个界面
# 计算残差的图像
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QUrl
from alive_progress import alive_bar
import time
from option import args  # 超分模型、baseline等所需的参数配置
# 导入自己的超分模型
from arbrcan import ArbRCAN  # 超分模型类（使用 ArbRCAN 网络）
# 导入baseline模型
from baseline import BaseLine  # baseline模型类

import utility  # 工具函数集合
from utility import make_coord  # 坐标生成函数，用于超分模型输入
import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib
import matplotlib.image as mpimg
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QMovie
from PyQt5.QtWidgets import QGraphicsScene, QFileDialog, QMessageBox, \
    QDesktopWidget, QDialog, QVBoxLayout
from PyQt5.QtWidgets import QLabel, QWidget, QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import QWebEngineView
import matplotlib.pyplot as plt
# 导入UI界面
from qt.mainWindow_v13 import Ui_MainWindow  # 主窗口的UI
from qt.login import Ui_Login  # 登录窗口的UI
import torch

from qt.angle import Ui_angleChoice  # 投影角度选择对话框UI
from qt.save import Ui_saveChoice  # 保存选择对话框UI
import qdarkstyle  # 黑暗风格的样式
import pynvml  # NVIDIA显卡内存管理工具
import sqlite3

import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


########################################################################
# 登录界面类
########################################################################

class loginWindow(QWidget, Ui_Login):
    """
    登录窗口：
    1. 初始化时调用 setupUi 构造界面，并将窗口居中。
    2. 调用 __buildingDB 创建并初始化一个简单的 SQLite 数据库，用于存储用户名和密码。
    3. 绑定登录按钮的点击事件 __login_button，该方法从数据库验证账号密码。
    """

    def __init__(self):
        super(loginWindow, self).__init__()
        self.setupUi(self)  # 根据 Ui_Login 生成UI界面
        self.__center()  # 居中显示
        self.__buildingDB()  # 初始化数据库（如果数据库已存在，注意此处可能重复创建）
        self.username.setPlaceholderText('please enter username...')
        self.password.setPlaceholderText('please enter password...')
        self.loginButton.clicked.connect(self.__login_button)  # 绑定登录按钮事件

    def __login_button(self):

        """
        登录按钮处理：
        1. 从输入框读取用户名和密码；
        2. 查询数据库判断是否存在匹配记录；
        3. 如果验证成功，则显示主窗口（mainWindow 由外部创建）并关闭登录窗口；
        4. 否则提示错误并清空密码输入框。
        """
        conn = sqlite3.connect("./db01.db")
        cursor = conn.cursor()
        user_id = self.username.text()  # 获取账号
        password = self.password.text()  # 获取密码
        # SQL语句，判断数据库中是否拥有这账号和密码
        sql = 'select username, password from table01 where username=? and password=? '
        if user_id == '':
            QMessageBox.warning(self, 'Warning', 'Username can not be empty！')
            return None
        if password == '':
            QMessageBox.warning(self, 'Warning', 'Password can not be empty！')
            return None
        cursor.execute(sql, (user_id, password))
        data = cursor.fetchall()
        # 如果匹配到数据，则打开主窗口并关闭登录窗口
        if data:
            # mainWindow.showFullScreen()
            # 暂时不全屏显示，取消注释可全屏显示
            mainWindow.show()
            self.close()
        else:
            QMessageBox.critical(self, 'Error', 'Wrong password！')
            self.password.clear()
            return None

    def __center(self):
        """
        将窗口居中显示：
        1. 获取窗口的几何尺寸；
        2. 获取屏幕中心点，并将窗口移动到屏幕中心。
        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def __buildingDB(self):
        """
        创建并初始化数据库：
        1. 使用 SQLite 创建名为 db01.db 的数据库（如果不存在则创建）；
        2. 建表 table01（用户名和密码字段）；
        3. 插入一些示例数据，便于测试。

        如果你需要替换验证方法，可以修改这里的 SQL 查询或改用其他数据库。
        """
        conn = sqlite3.connect("./db01.db")
        cur = conn.cursor()
        # cur.execute("CREATE TABLE table01(username text,password text)")
        cur.execute("INSERT INTO table01 values('zhangsan','123')")
        cur.execute("INSERT INTO table01 values('lisi','456')")
        cur.execute("INSERT INTO table01 values('wangwu','789')")
        cur.execute("INSERT INTO table01 values('wangwu1','7890')")
        cur.execute("INSERT INTO table01 values('admin','1234')")
        conn.commit()
        cur.close()


########################################################################
# 主窗口类
########################################################################


class MainWindow(QMainWindow, Ui_MainWindow):
    """
    主窗口：
    1. 初始化时调用 setupUi 构造UI界面，并对窗口居中。
    2. 初始化变量：例如设备标志（用于超分处理时判断CUDA内存是否足够）、当前加载图片文件名等。
    3. 定义各个图像处理的中间变量和保存路径（例如GT、temp、save目录）。
    4. 绑定菜单和按钮的点击事件，如打开文件、保存文件、调用拉东变换、超分模型、归因图、反拉东变换等。
    5. 各个方法中调用的系统命令（如 os.system 调用脚本）和模型调用方法也均在此类中定义。

    如果要替换某个方法（例如超分模型的实现），你可以修改 __ourModel 或 __baseModel 方法，或者绑定其他函数。
    """

    def __init__(self):
        super(MainWindow, self).__init__()
        # 在MainWindow的初始化方法或程序启动时添加
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.empty_cache()
        self.setupUi(self)  # 根据 Ui_MainWindow 构造UI
        self.__center()  # 居中显示
        self.deviceFlag = False  # 标志是否找到了可用的CUDA设备
        self.__lamWidow = None  # 用于归因图窗口的实例

        self.hover_effect_style = """
            QGraphicsView {
                border: 2px solid #00FF00;
                background-color: rgba(0, 255, 0, 50);
            }
        """
        self.normal_style = "QGraphicsView { border: 1px solid gray; }"
        self.selected_style = """
            QGraphicsView {
                border: 3px solid #FF0000;
            }
        """
        # 初始时将所有视图设置为正常样式
        self.radonView.setStyleSheet(self.normal_style)  # 窗口 1：正弦图
        self.iradonView.setStyleSheet(self.normal_style)  # 窗口 2：FBP
        self.srcImageView.setStyleSheet(self.normal_style)  # 窗口 3：无预训练 INR
        self.srView.setStyleSheet(self.normal_style)  # 窗口 4：预训练 INR

        self.radonView.installEventFilter(self)  # 正弦图
        self.iradonView.installEventFilter(self)  # FBP
        self.srcImageView.installEventFilter(self)  # 无预训练 INR
        self.srView.installEventFilter(self)  # 预训练 INR


        self.Fname = None  # 当前打开的图片文件名
        self.t_Fname = None  # 临时图片文件名
        self.selected_fname = None
        # 初始化归因图参数及角度设置
        # x和y表示某个点图像区域的坐标
        self.x = 0
        self.y = 0
        self.wis = 0
        # 默认拉东变换的坐标是180度
        self.radonAngle = 180
        # 设置图片存放路径，用于加载图片进行处理
        self.projDir = './proj/'  # 原图目录
        self.gtDir = './ct_gt/'
        self.tempDir = './temp/'  # 中间文件目录
        self.saveDir = './save/'  # 保存目录
        self.reconDir = './recon/'
        # 定义具体的文件名
        self.gtName = 'GT.png'
        self.reconName = self.reconDir + 'recon.png'
        # 超分后得到的sinogram
        # 修改成员变量命名（可选）
        self.srSinoName = self.tempDir + 'nearSino.png'
        # baseline模型处理后得到的正弦图
        self.baseSinoName = self.tempDir + 'baseSino.png'

        # nearSinoName、bilSinoName、bicSinoName 分别用于保存最近邻、双线性、双三次插值后的图像。
        self.nearSinoName = self.tempDir + 'nearSino.png'
        self.bilSinoName = self.tempDir + 'bilSino.png'
        self.bicSinoName = self.tempDir + 'bicSino.png'


        # 定义四个窗口的图像路径变量
        self.sinogram_path = None  # 第一个窗口：正弦图路径
        self.fbp_recon_path = None  # 第二个窗口：FBP 重建结果路径
        self.nopre_recon_path = None  # 第三个窗口：无预训练 INR 重建结果路径
        self.pre_recon_path = None  # 第四个窗口：预训练 INR 重建结果路径
        self.current_gt_path = None  # 对应的真实图像（GT）路径
        # 存储用户所选图像的路径（最多允许 3 个选择）
        self.selected_images = []  # 列表用于保存所选图像的路径



        # self.angle.setValue(180)  # 初始化角度控件为180°

        # 绑定各菜单和按钮事件：
        # 文件菜单：打开文件、保存文件、退出程序等
        self.openFileAction.triggered.connect(self.__openFileAndShowImage)
        self.openlitt.clicked.connect(self.__openFileAndShowImage)
        self.saveFileAction.triggered.connect(self.saveFile)
        self.saveOrigin.clicked.connect(self.saveOri)
        self.saverandon.clicked.connect(self.saveran)
        self.saveSr.clicked.connect(self.saveSr1)
        self.saveIradon.clicked.connect(self.saveIra)
        # 绑定退出动作，点击后调用close（）方法
        self.exitAppAction.triggered.connect(self.close)
        self.closeButton.clicked.connect(self.close)
        # 拉东变换菜单，选择这个菜单的时候，调用self.__radonImage方法
        self.actionRadon.triggered.connect(self.__radonImage)
        # 超分模型菜单（调用自定义的超分模型）
        self.actionModel.triggered.connect(self.__srImage)
        # # baseline模型菜单
        self.actionBaseline.triggered.connect(self.__baseImage)
        # 三种插值算法菜单
        self.actionNearest_neighbor.triggered.connect(self.__nearestImage)
        self.actionBilinear.triggered.connect(self.__bilinearImage)
        self.actionBicubic.triggered.connect(self.__bicubicImage)
        # 归因图菜单，绑定归因图处理，调用_lamImage方法
        self.actionLAM.triggered.connect(self.__lamImage)
        # 反拉东变换菜单
        self.actionFBP.triggered.connect(self.__fbpRecon)


        # 绑定SSIM和PSNR按钮
        # self.SSIM_button.clicked.connect(self.__show_ssim_analysis)
        # self.PSNR3D_button.clicked.connect(self.__show_psnr_analysis)
        self.plot_windows = []

        self.clear.clicked.connect(self.__clearAllImages)

        # 绑定 Pre_Nerp 按钮点击事件
        self.actionPre_Nerp.triggered.connect(self.show_image_in_pretraining)
        # 绑定 NoPre_Nerp 按钮点击事件
        self.actionNoPre_Nerp.triggered.connect(self.show_image_in_training)


        self.SSIM_button.clicked.connect(self.handle_analysis)
        # self.PSNR3D_button.clicked.connect(self.__show_psnr_analysis)

    def handle_analysis(self):
        if not self.selected_images:
            QMessageBox.warning(self, "警告", "请先选择至少一张图像进行分析")
            return
        if not self.current_gt_path or not os.path.exists(self.current_gt_path):
            QMessageBox.warning(self, "警告", "无法找到对应的 GT 图像")
            return

        for target_path in self.selected_images:
            if not os.path.exists(target_path):
                QMessageBox.warning(self, "警告", f"图像路径无效: {target_path}")
                continue

            # Execute SSIM analysis and generate result
            analysis = PlotAnalysis(self.current_gt_path, target_path)
            html_path = analysis.generate_ssim_plot()
            self.__show_plot_window(html_path)


    def __show_plot_window(self, html_path):
        """在新窗口中显示分析结果"""
        plot_win = PlotWindow(html_path)
        plot_win.show()
        self.plot_windows.append(plot_win)  # 保留引用以避免窗口被垃圾回收

    def eventFilter(self, watched, event):
        # Define the views that can be selected
        affected_views = [self.radonView, self.srView, self.iradonView]

        # Mouse enter event (hover effect)
        if event.type() == QtCore.QEvent.Enter and watched in affected_views:
            watched.setStyleSheet(self.hover_effect_style)

        # Mouse leave event (reset to normal if not selected)
        elif event.type() == QtCore.QEvent.Leave and watched in affected_views:
            if watched.styleSheet() != self.selected_style:
                watched.setStyleSheet(self.normal_style)

        # Mouse click event (selection)
        elif event.type() == QtCore.QEvent.MouseButtonPress and watched in affected_views:
            # Reset all affected views to normal style
            for view in affected_views:
                view.setStyleSheet(self.normal_style)
            # Set the clicked view to selected style
            watched.setStyleSheet(self.selected_style)
            # Determine the path based on the clicked view
            if watched == self.radonView:
                path = self.fbp_recon_path
                type_name = "FBP"
            elif watched == self.srView:
                path = self.nopre_recon_path
                type_name = "无预训练 INR"
            elif watched == self.iradonView:
                path = self.pre_recon_path
                type_name = "预训练 INR"
            # Set selected_images to contain only this path
            self.selected_images = [path]
            self.statusBar().showMessage(f"已选择 {type_name} 图像进行分析: {path}", 3000)

        return super().eventFilter(watched, event)


    def parse_gt_path(self, sino_path):
        """从正弦图文件名提取 GT 路径（例如，img1_proj45.png -> img_1.png）"""
        basename = os.path.basename(sino_path)
        match = re.match(r'img(\d+)_proj\d+\.', basename)
        if match:
            img_num = match.group(1)
            return os.path.join(self.gtDir, f"img_{img_num}.png")
        return None


    def __openFileAndShowImage(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open File', self.projDir,
                                               "Picture Files(*.png *.jpg *.jpeg *.gif *.bmp)")
        if fname:
            self.selected_fname = fname
            self.Fname = fname
            self.sinogram_path = fname  # 分配正弦图路径
            self.current_gt_path = self.parse_gt_path(fname)  # 映射到对应的 GT 路径
            self.__loadImage(fname, window=1)
        else:
            QMessageBox.information(self, 'Notice', '未选择任何文件！')


    def __fbpRecon(self):
        """从第一个窗口的正弦图直接进行FBP重建"""
        if not hasattr(self, 'Fname'):
            QMessageBox.warning(self, "Warning", "请先加载正弦图")
            return

        # 读取正弦图数据
        sino_img = cv2.imread(self.Fname, cv2.IMREAD_GRAYSCALE)

        # 执行FBP重建（使用修改后的方法）
        QMessageBox.information(self, "Notice", "重建中，请稍候...")
        recon_result = self.reconFBP(sino_img, img_size=512)  # 根据实际尺寸调整

        # 保存并显示重建结果
        save_path = os.path.join(self.saveDir, "recon_result.png")
        cv2.imwrite(save_path, recon_result)
        self.fbp_recon_path = save_path  # 分配 FBP 重建结果路径
        self.__loadImage(save_path, window=2)


    def show_image_in_pretraining(self):
        """动态显示预训练重建序列到 traing 框（带渐变效果）"""
        try:
            if self.selected_fname:
                # 同样从 selected_fname 提取 img 序号和 proj 角度
                basename = os.path.basename(self.selected_fname)
                m = re.match(r'img(\d+)_proj(\d+)\.\w+$', basename)
                if not m:
                    QMessageBox.critical(self, "格式错误",
                                         f"文件名不符合 img<序号>_proj<角度> 格式：\n{basename}")
                    return
                img_num, angle = m.group(1), m.group(2)
                base_dir = os.path.join("Pre-INR", f"proj{angle}", f"recon_img{img_num}")

            if not os.path.exists(base_dir):
                QMessageBox.critical(self, "路径错误", f"目录不存在：\n{base_dir}")
                return

            # 2. 重置动画状态
            # 如果之前有动画组，先断开它的 finished 信号
            if hasattr(self, 'animation_group'):
                try:
                    self.animation_group.finished.disconnect()
                except TypeError:
                    pass
            self.animation_group = QtCore.QParallelAnimationGroup(self)
            self.animation_running = False
            self.previous_pixmap = None
            self.current_image_index = 0

            # 3. 收集并排序图片
            image_files = []
            for f in os.listdir(base_dir):
                if f.startswith("recon_") and f.endswith(".png"):
                    try:
                        seq = int(f.split('_')[1])
                        image_files.append((seq, os.path.join(base_dir, f)))
                    except Exception:
                        continue
            if not image_files:
                QMessageBox.critical(self, "数据错误", "未找到有效图片文件")
                return
            image_files.sort(key=lambda x: x[0])
            self.image_sequence = [p for _, p in image_files[:11]]

            # 4. 代理类，用于动画控制 opacity
            class PixmapItemProxy(QtCore.QObject):
                opacityChanged = QtCore.pyqtSignal(float)

                def __init__(self, item):
                    super().__init__()
                    self._item = item
                    self._opacity = 1.0

                @QtCore.pyqtProperty(float)
                def opacity(self):
                    return self._opacity

                @opacity.setter
                def opacity(self, v):
                    self._opacity = v
                    self._item.setOpacity(v)
                    self.opacityChanged.emit(v)

            # 5. 动画切换函数
            def animate_transition(new_pixmap):
                self.animation_group.clear()
                scene = QGraphicsScene()

                # 旧图淡出
                if self.previous_pixmap:
                    old_item = scene.addPixmap(self.previous_pixmap)
                    old_proxy = PixmapItemProxy(old_item)
                    fade_out = QtCore.QPropertyAnimation(old_proxy, b"opacity")
                    fade_out.setDuration(800)
                    fade_out.setStartValue(1.0)
                    fade_out.setEndValue(0.0)
                    fade_out.setEasingCurve(QtCore.QEasingCurve.OutQuad)
                    self.animation_group.addAnimation(fade_out)

                # 新图淡入
                new_item = scene.addPixmap(new_pixmap)
                new_item.setOpacity(0.0)
                new_proxy = PixmapItemProxy(new_item)
                fade_in = QtCore.QPropertyAnimation(new_proxy, b"opacity")
                fade_in.setDuration(1000)
                fade_in.setStartValue(0.0)
                fade_in.setEndValue(1.0)
                fade_in.setEasingCurve(QtCore.QEasingCurve.InQuad)
                self.animation_group.addAnimation(fade_in)

                # 更新视图
                self.traing.setScene(scene)
                self.traing.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

            # 6. 递归加载下一张
            def load_next_image():
                # 如果动画正在运行，跳过
                if self.animation_running:
                    return

                # 到达末尾：显示最终图并返回
                if self.current_image_index >= len(self.image_sequence):
                    final_path = self.image_sequence[-1]
                    try:
                        img = cv2.imread(final_path, cv2.IMREAD_UNCHANGED)
                        if img.dtype != np.uint8:
                            img = cv2.normalize(img, None, 0, 255,
                                                cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        temp_path = os.path.join(self.tempDir, "pre_final_display.png")
                        cv2.imwrite(temp_path, img)
                        self.pre_recon_path = temp_path

                        self.reconName = os.path.join(self.reconDir, "recon.png")
                        cv2.imwrite(self.reconName, img)
                        self.__loadImage(temp_path, window=4)

                    except Exception as e:
                        QMessageBox.critical(self, "图像处理错误", f"最终图保存失败：{e}")
                    return

                # 标记动画开始
                self.animation_running = True

                # 读取当前图
                path = self.image_sequence[self.current_image_index]
                pixmap = QPixmap(path)
                if pixmap.isNull():
                    QMessageBox.critical(self, "加载失败", f"无法加载：{path}")
                    self.animation_running = False
                    return

                # 执行渐变动画
                animate_transition(pixmap)
                self.previous_pixmap = pixmap
                self.statusBar().showMessage(
                    f"重建进度：{self.current_image_index + 1}/{len(self.image_sequence)}", 2500
                )

                # 连接动画完成信号，并在回调中断开，防止重复
                def on_finish():
                    # 恢复运行状态并断开信号
                    self.animation_running = False
                    self.current_image_index += 1
                    try:
                        self.animation_group.finished.disconnect(on_finish)
                    except TypeError:
                        pass
                    # 下一张
                    QtCore.QTimer.singleShot(500, load_next_image)

                self.animation_group.finished.connect(on_finish, Qt.UniqueConnection)
                self.animation_group.start()

            # 7. 启动
            load_next_image()

        except Exception as e:
            QMessageBox.critical(self, "运行时错误", f"初始化失败：{e}")

    def show_image_in_training(self):
        """动态显示预训练重建序列到 traing 框（带渐变效果）"""
        try:
            if self.selected_fname:
                # 同样从 selected_fname 提取 img 序号和 proj 角度
                basename = os.path.basename(self.selected_fname)
                m = re.match(r'img(\d+)_proj(\d+)\.\w+$', basename)
                if not m:
                    QMessageBox.critical(self, "格式错误",
                                         f"文件名不符合 img<序号>_proj<角度> 格式：\n{basename}")
                    return
                img_num, angle = m.group(1), m.group(2)
                base_dir = os.path.join("NoPre-INR", f"proj{angle}", f"recon_img{img_num}")

            if not os.path.exists(base_dir):
                QMessageBox.critical(self, "路径错误", f"目录不存在：\n{base_dir}")
                return

            # 2. 重置动画状态
            # 如果之前有动画组，先断开它的 finished 信号
            if hasattr(self, 'animation_group'):
                try:
                    self.animation_group.finished.disconnect()
                except TypeError:
                    pass
            self.animation_group = QtCore.QParallelAnimationGroup(self)
            self.animation_running = False
            self.previous_pixmap = None
            self.current_image_index = 0

            # 3. 收集并排序图片
            image_files = []
            for f in os.listdir(base_dir):
                if f.startswith("recon_") and f.endswith(".png"):
                    try:
                        seq = int(f.split('_')[1])
                        image_files.append((seq, os.path.join(base_dir, f)))
                    except Exception:
                        continue
            if not image_files:
                QMessageBox.critical(self, "数据错误", "未找到有效图片文件")
                return
            image_files.sort(key=lambda x: x[0])
            self.image_sequence = [p for _, p in image_files[:11]]

            # 4. 代理类，用于动画控制 opacity
            class PixmapItemProxy(QtCore.QObject):
                opacityChanged = QtCore.pyqtSignal(float)

                def __init__(self, item):
                    super().__init__()
                    self._item = item
                    self._opacity = 1.0

                @QtCore.pyqtProperty(float)
                def opacity(self):
                    return self._opacity

                @opacity.setter
                def opacity(self, v):
                    self._opacity = v
                    self._item.setOpacity(v)
                    self.opacityChanged.emit(v)

            # 5. 动画切换函数
            def animate_transition(new_pixmap):
                self.animation_group.clear()
                scene = QGraphicsScene()

                # 旧图淡出
                if self.previous_pixmap:
                    old_item = scene.addPixmap(self.previous_pixmap)
                    old_proxy = PixmapItemProxy(old_item)
                    fade_out = QtCore.QPropertyAnimation(old_proxy, b"opacity")
                    fade_out.setDuration(800)
                    fade_out.setStartValue(1.0)
                    fade_out.setEndValue(0.0)
                    fade_out.setEasingCurve(QtCore.QEasingCurve.OutQuad)
                    self.animation_group.addAnimation(fade_out)

                # 新图淡入
                new_item = scene.addPixmap(new_pixmap)
                new_item.setOpacity(0.0)
                new_proxy = PixmapItemProxy(new_item)
                fade_in = QtCore.QPropertyAnimation(new_proxy, b"opacity")
                fade_in.setDuration(1000)
                fade_in.setStartValue(0.0)
                fade_in.setEndValue(1.0)
                fade_in.setEasingCurve(QtCore.QEasingCurve.InQuad)
                self.animation_group.addAnimation(fade_in)

                # 更新视图
                self.traing.setScene(scene)
                self.traing.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

            # 6. 递归加载下一张
            def load_next_image():
                # 如果动画正在运行，跳过
                if self.animation_running:
                    return

                # 到达末尾：显示最终图并返回
                if self.current_image_index >= len(self.image_sequence):
                    final_path = self.image_sequence[-1]
                    try:
                        img = cv2.imread(final_path, cv2.IMREAD_UNCHANGED)
                        if img.dtype != np.uint8:
                            img = cv2.normalize(img, None, 0, 255,
                                                cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        temp_path = os.path.join(self.tempDir, "NoPrefinal_display.png")

                        cv2.imwrite(temp_path, img)
                        self.nopre_recon_path = temp_path

                        self.reconName = os.path.join(self.reconDir, "recon.png")
                        cv2.imwrite(self.reconName, img)

                        self.__loadImage(temp_path, window=3)
                    except Exception as e:
                        QMessageBox.critical(self, "图像处理错误", f"最终图保存失败：{e}")
                    return

                # 标记动画开始
                self.animation_running = True

                # 读取当前图
                path = self.image_sequence[self.current_image_index]
                pixmap = QPixmap(path)
                if pixmap.isNull():
                    QMessageBox.critical(self, "加载失败", f"无法加载：{path}")
                    self.animation_running = False
                    return

                # 执行渐变动画
                animate_transition(pixmap)
                self.previous_pixmap = pixmap
                self.statusBar().showMessage(
                    f"重建进度：{self.current_image_index + 1}/{len(self.image_sequence)}", 2500
                )

                # 连接动画完成信号，并在回调中断开，防止重复
                def on_finish():
                    # 恢复运行状态并断开信号
                    self.animation_running = False
                    self.current_image_index += 1
                    try:
                        self.animation_group.finished.disconnect(on_finish)
                    except TypeError:
                        pass
                    # 下一张
                    QtCore.QTimer.singleShot(500, load_next_image)

                self.animation_group.finished.connect(on_finish, Qt.UniqueConnection)
                self.animation_group.start()

            # 7. 启动
            load_next_image()

        except Exception as e:
            QMessageBox.critical(self, "运行时错误", f"初始化失败：{e}")

    def __clearAllImages(self):
        try:
            # 假设 self.image_view1 到 self.image_view4 是用于显示图像的 QGraphicsView 对象
            for view in [self.traing, self.srcImageView, self.srView, self.radonView,self.iradonView]:
                scene = QGraphicsScene()
                view.setScene(scene)  # 清空场景
                view.update()  # 更新视图
        except Exception as e:
            print(f"清除图像时发生错误: {e}")

    def __show_ssim_analysis(self):
        """显示SSIM分析结果"""
        if not self.Fname or not os.path.exists(self.reconName):
            QMessageBox.warning(self, "错误", "请先加载原始图像并完成重建")
            return

        plot_analysis = PlotAnalysis(self.Fname, self.reconName)
        html_path = plot_analysis.generate_ssim_plot()
        self.__show_plot_window(html_path)

    # def __show_psnr_analysis(self):
    #     """显示PSNR 3D分析结果"""
    #     if not self.Fname or not os.path.exists(self.reconName):
    #         QMessageBox.warning(self, "错误", "请先加载原始图像并完成重建")
    #         return
    #
    #     plot_analysis = PlotAnalysis(self.Fname, self.reconName)
    #     html_path = plot_analysis.generate_psnr_3d_plot()
    #     self.__show_plot_window(html_path)

    def __show_plot_window(self, html_path):
        """显示图表窗口"""
        if not os.path.exists(html_path):
            QMessageBox.critical(self, "错误", "图表文件生成失败")
            return

        plot_win = PlotWindow(html_path)
        plot_win.show()
        self.plot_windows.append(plot_win)  # 保存PlotWindow对象

    # 重建显示
    def __reconstructImage(self):
        if self.radonView.items() == []:
            QMessageBox.information(self, 'Notice', '请先生成正弦图！')
            return

        # 调用你的重建模型（示例）
        QMessageBox.information(self, 'Notice', '正在重建，请稍候...')
        sino_path = self.sinoName  # 正弦图路径
        output_path = self.srSinoName

        # 这里替换为你的重建模型调用（示例用系统命令）
        ret = os.system(f'python your_recon_model.py {sino_path} {output_path}')

        if ret == 0:
            self.__loadImage(output_path, window=3)  # 在第三个窗口显示重建结果
            self.__showDiffAnalysis()  # 自动跳转到差异分析
        else:
            QMessageBox.critical(self, 'Error', '重建失败！')

    # -------------------------寻找内存充足的CUDA设备 ---------------------------
    def __searchDevice(self):

        """
        通过 pynvml 检查所有CUDA设备，找出至少有 10000 MB 空闲内存的设备，
        并将 os.environ['CUDA_VISIBLE_DEVICES'] 设置为该设备索引。
        如果你需要替换显卡选择逻辑，可在此处修改判断条件。
        """

        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        for id in range(deviceCount):
            handler = pynvml.nvmlDeviceGetHandleByIndex(id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
            free = round(meminfo.free / 1024 / 1024, 2)
            # 找出所有至少有10000MB的设备
            if free >= 10000:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(id)
                self.device = 'cuda'
                self.deviceFlag = True

    # -------------------------窗口居中 ---------------------------
    def __center(self):
        """
        将当前窗口居中显示（同前述方法）。
        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # -------------------------加载并显示图像 ---------------------------
    def __loadImage(self, fname, window):
        """
        加载图像并保存至对应窗口显示：
        根据传入的 window 参数，区分不同用途（1：原图；2：拉东变换图；3：超分/插值结果；4：反拉东重建图）。
        读取图像后调用 __displayImage 实际显示。
        """
        # self.selected_fname = fname
        if window == 4:
            self.image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            self.window = window
            self.tmp = self.image  # 存原图
            self.__displayImage(fname)
        else:
            self.image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            self.t_Fname = fname
            self.window = window
            self.tmp = self.image
            self.__displayImage(fname)

    def __displayImage(self, name):
        """
        将 cv2 读取的图像转换成 QImage，再转换成 QPixmap，
        并放入 QGraphicsScene 内，最后设置到对应的 QGraphicsView 显示。
        根据 self.window 参数设置不同区域的文本、对齐方式等。
        """
        if name is None:
            # 清空 QGraphicsScene
            scene = QGraphicsScene()
            # 假设你有一个 QGraphicsView 对象，这里用 self.graphicsView 代替
            self.graphicsView.setScene(scene)
        else:
            qformat = QImage.Format_Indexed8
            if len(self.image.shape) == 3:
                if (self.image.shape[2]) == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888
            img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
            img = img.rgbSwapped()  # RGB与BGR转换
            __qPixmap = QPixmap.fromImage(img)
            __scene = QGraphicsScene()
            text = 'Length:{} Width:{}'
            __scene.addPixmap(__qPixmap)
            if self.window == 1:
                self.src_name.setText(self.t_Fname.split('/')[-1].split('.')[0])
                self.src_label.setText(text.format(self.image.shape[1], self.image.shape[0]))
                self.srcImageView.setScene(__scene)
                self.srcImageView.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            if self.window == 2:
                self.sino_name.setText(self.t_Fname.split('/')[-1].split('.')[0])
                self.radon_label.setText(text.format(self.image.shape[1], self.image.shape[0]))
                self.radonView.setScene(__scene)
                self.radonView.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            if self.window == 3:
                self.sr_label.setText(text.format(self.image.shape[1], self.image.shape[0]))
                self.srView.setScene(__scene)
                self.srView.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            if self.window == 4:
                self.recon_name.setText(name.split('/')[-1].split('.')[0])
                self.iradon_label.setText(text.format(self.image.shape[1], self.image.shape[0]))
                self.iradonView.setScene(__scene)
                self.iradonView.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


    # -------------------------保存文件相关方法 ---------------------------
    def saveFile(self):
        """
        弹出保存选择对话框（saveWindow），根据对话框返回的 index 决定保存哪一类图像。
        如果需要更换保存逻辑，可以在 saveIm 方法中修改。
        """
        self.saveWin = saveWindow()
        self.saveWin.show()
        self.saveWin.index.connect(self.saveIm)

    def saveOri(self):
        if self.srcImageView.items() == []:
            QMessageBox.information(self, 'Notice', 'File save failed！')
        else:
            fname, _ = QFileDialog.getSaveFileName(self, 'Save File', self.saveDir,
                                                   "Picture Files(*.png *.jpg *.jpeg *.gif *.bmp)")
            if fname:
                current_img = cv2.imread(self.projDir + self.src_name.text() + '.png')
                cv2.imwrite(fname, current_img)
            else:
                QMessageBox.information(self, 'Notice', 'File save failed！')

    def saveran(self):
        if self.radonView.items() == []:
            QMessageBox.information(self, 'Notice', 'File save failed！')
        else:
            fname, _ = QFileDialog.getSaveFileName(self, 'Save File', self.saveDir,
                                                   "Picture Files(*.png *.jpg *.jpeg *.gif *.bmp)")
            if fname:
                print(fname)
                current_img = cv2.imread(self.tempDir + self.sino_name.text() + '.png')
                cv2.imwrite(fname, current_img)
            else:
                QMessageBox.information(self, 'Notice', 'File save failed！')

    def saveSr1(self):
        if self.srView.items() == []:
            QMessageBox.information(self, 'Notice', 'File save failed！')
        else:
            fname, _ = QFileDialog.getSaveFileName(self, 'Save File', self.saveDir,
                                                   "Picture Files(*.png *.jpg *.jpeg *.gif *.bmp)")
            if fname:
                current_img = cv2.imread(self.tempDir + self.sr_name.text() + '.png')
                cv2.imwrite(fname, current_img)
            else:
                QMessageBox.information(self, 'Notice', 'File save failed！')

    def saveIra(self):
        if self.iradonView.items() == []:
            QMessageBox.information(self, 'Notice', 'File save failed！')
        else:
            fname, _ = QFileDialog.getSaveFileName(self, 'Save File', self.saveDir,
                                                   "Picture Files(*.png *.jpg *.jpeg *.gif *.bmp)")
            if fname:
                current_img = cv2.imread(self.recon_name.text() + '.png')
                cv2.imwrite(fname, current_img)
            else:
                QMessageBox.information(self, 'Notice', 'File save failed！')

    def saveIm(self, index):
        """
        根据 index 决定保存哪种图像：
        index = 0：原图
        index = 1：拉东图
        index = 2：超分/插值结果图
        index = 3：重建图（反拉东）
        """
        if index == 0:
            if self.srcImageView.items() == []:
                QMessageBox.information(self, 'Notice', 'File save failed！')
            else:
                fname, _ = QFileDialog.getSaveFileName(self, 'Save File', self.saveDir,
                                                       "Picture Files(*.png *.jpg *.jpeg *.gif *.bmp)")
                if fname:
                    current_img = cv2.imread(self.projDir + self.src_name.text() + '.png')
                    cv2.imwrite(fname, current_img)
                else:
                    QMessageBox.information(self, 'Notice', 'File save failed！')
        elif index == 1:
            if self.radonView.items() == []:
                QMessageBox.information(self, 'Notice', 'File save failed！')
            else:
                fname, _ = QFileDialog.getSaveFileName(self, 'Save File', self.saveDir,
                                                       "Picture Files(*.png *.jpg *.jpeg *.gif *.bmp)")
                if fname:
                    print(fname)
                    current_img = cv2.imread(self.tempDir + self.sino_name.text() + '.png')
                    cv2.imwrite(fname, current_img)
                else:
                    QMessageBox.information(self, 'Notice', 'File save failed！')
        elif index == 2:
            if self.srView.items() == []:
                QMessageBox.information(self, 'Notice', 'File save failed！')
            else:
                fname, _ = QFileDialog.getSaveFileName(self, 'Save File', self.saveDir,
                                                       "Picture Files(*.png *.jpg *.jpeg *.gif *.bmp)")
                if fname:
                    current_img = cv2.imread(self.tempDir + self.sr_name.text() + '.png')
                    cv2.imwrite(fname, current_img)
                else:
                    QMessageBox.information(self, 'Notice', 'File save failed！')
        elif index == 3:
            if self.iradonView.items() == []:
                QMessageBox.information(self, 'Notice', 'File save failed！')
            else:
                fname, _ = QFileDialog.getSaveFileName(self, 'Save File', self.saveDir,
                                                       "Picture Files(*.png *.jpg *.jpeg *.gif *.bmp)")
                if fname:
                    current_img = cv2.imread(self.recon_name.text() + '.png')
                    cv2.imwrite(fname, current_img)
                else:
                    QMessageBox.information(self, 'Notice', 'File save failed！')
        else:
            QMessageBox.information(self, 'Notice', 'File save failed！')

    # -------------------------重写关闭事件 ---------------------------
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        """
        当主窗口关闭时，调用 sys.exit(0) 退出程序，确保所有子窗口一并关闭。
        """
        sys.exit(0)

    # -------------------------拉东变换相关 ---------------------------
    def __radonImage(self):
        """
        拉东变换操作：
        1. 检查是否已经加载图片；
        2. 弹出投影角度选择对话框（angleWindow），用户选择后触发 angleChanged。
        """
        if self.Fname:
            self.angleWinow = angleWindow()
            self.angleWinow.show()
            # 显示选择角度文本框
            self.angleWinow.index.connect(self.angleChanged)
        else:
            QMessageBox.information(self, 'Notice', 'Missing image！')

    def radon_GT(self):
        """
        通过调用自定义 GetProj 函数生成 GT 图像的投影数据（正弦图），
        并将归一化后的图像保存为 self.gtName 指定的 PNG 文件。
        """
        # 这里取投影数量为180，可以根据需要调整
        num_proj = 180
        # 调用自定义函数生成投影数据，返回一个 numpy 数组
        projs = self.GetProj(num_proj, self.Fname)

        # 将投影数据归一化到0-255（uint8）
        projs_norm = ((projs - np.min(projs)) / (np.max(projs) - np.min(projs)) * 255).astype(np.uint8)

        # 保存归一化后的图像到 GT 文件路径（self.gtName）
        cv2.imwrite(self.gtName, projs_norm)


    # 前半截做一个正弦图的插值，后面做一个在插值基础上的重建
    def GetProj(sefl, num_proj, img_name):
        """
        读取 PNG 格式的灰度图像文件 img_name，
        并利用 ODL 生成投影数据（正弦图）。

        参数：
          num_proj: 投影数量（决定角度采样）
          img_name: PNG 图像文件路径

        返回：
          projs: 生成的正弦图数据（numpy 数组）
        """
        img = Image.open(img_name)  # 不会转灰度，保持原始格式
        img = np.array(img)
        if img is None:
            raise ValueError(f"无法加载图像：{img_name}")

        # 直接将图像宽度作为 detector 的采样数
        num_dete = img.shape[1]

        # 构造重构空间，使用图像的尺寸（高度、宽度）作为采样形状
        reco_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1],
                                       shape=img.shape, dtype='float32')

        # 生成投影角度，均匀分布在 [0, π) 内
        theta = np.linspace(0, np.pi, num_proj, endpoint=False)
        grid = odl.RectGrid(theta)
        angles = odl.uniform_partition_fromgrid(grid)

        # 定义检测器分割，采样数量采用图像宽度 num_dete
        detector_partition = odl.uniform_partition(-1, 1, num_dete)

        # 构造平行射线几何模型
        geometry = odl.tomo.Parallel2dGeometry(angles, detector_partition)

        # 创建射线变换对象，这里采用 ASTRA CUDA 后端（根据你的环境，也可以用 astra_cpu）
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

        # ray_trafo.adjoint.domain.space.impl.astra_plugin.astra.clear()
        # 计算投影数据（正弦图）
        projs = np.array(ray_trafo(img))

        return projs

    def radon_image(self, index):
        """
        根据选择的角度（index）调用你自己写的函数 GetProj 来生成对应的 sinogram 图片。
        index=0：180°、1：90°、2：60°、3：45°。
        返回 True 表示成功。
        """
        # 映射 index 到投影数量及对应的角度
        if index == 0:
            num_proj = 180
            self.radonAngle = 180
        elif index == 1:
            num_proj = 55
            self.radonAngle = 55
        elif index == 2:
            num_proj = 45
            self.radonAngle = 45
        elif index == 3:
            num_proj = 35
            self.radonAngle = 35
        else:
            # 非法的角度索引
            return False

        # 构造保存的 sinogram 图片路径
        self.sinoName = self.tempDir + 'sino_{}.png'.format(self.radonAngle)

        # 调用自定义 GetProj 函数生成正弦图（投影）
        projs = self.GetProj(num_proj, self.Fname)

        # 保存生成的 sinogram 图片（这里用 matplotlib 保存图像为灰度图）
        plt.imsave(self.sinoName, projs, cmap='gray')
        # 你也可以在此处调用 __loadImage() 将生成的图片加载到第二个窗口进行显示
        self.__loadImage(self.sinoName, window=2)

        return True

    # def angleChanged(self, index):
    #
    #     """
    #     投影角度选择对话框返回后：
    #     1. 弹出提示等待时间；
    #     2. 利用线程池并行调用生成 GT 图和拉东图；
    #     3. 当任务完成后，加载并显示生成的 sinogram 图像到窗口2。
    #     """
    #     QMessageBox.information(self, 'Notice', 'It will take about 10 seconds, please be patient！')
    #     pool = ThreadPoolExecutor(max_workers=2)
    #     pool.submit(self.radon_GT)
    #     future2 = pool.submit(self.radon_image(index))
    #     if future2.done():
    #         self.__loadImage(self.sinoName, window=2)

    # 修改angleChanged方法中的线程池逻辑
    def angleChanged(self, index):
        QMessageBox.information(self, 'Notice', 'It will take about 10 seconds, please be patient！')
        # 先执行radon_GT，完成后执行radon_image
        self.radon_GT()
        success = self.radon_image(index)
        if success:
            self.__loadImage(self.sinoName, window=2)

    def angleChanged_v1(self):

        """
        一键处理时，依据角度控件的值执行拉东变换：
        将角度映射到对应的脚本索引，并并行调用生成GT图和拉东图，最后加载结果。
        """

        QMessageBox.information(self, 'Notice', 'It will take about 10 seconds, please be patient！')
        pool = ThreadPoolExecutor(max_workers=2)
        pool.submit(self.radon_GT)
        angle_index_map = {180: 0, 55: 1, 45: 2, 35: 3}
        angle = self.angle.value()
        if angle not in angle_index_map:
            QMessageBox.warning(self, "Angle Error", "Invalid angle value")
            return
        index = angle_index_map[angle]
        future2 = pool.submit(self.radon_image(index))
        if future2.done():
            self.__loadImage(self.sinoName, window=2)

    # -------------------------超分与baseline模型调用 ---------------------------

    def __ourModel(self, sino_name):

        """
        调用超分模型：
        1. 根据输入 sinogram 图像的尺寸选择对应的预训练模型（theta=90/60/45）；
        2. 将图像转换为Tensor，设置模型放大比例；
        3. 调用模型生成高分辨率图像，并进行量化处理。

        如果需要替换超分方法，可修改此函数中模型的加载和处理逻辑。
        """

        device = self.device
        model = ArbRCAN(args).to(device)
        low_res = cv2.imread(sino_name)
        if low_res.shape[1] == 90:
            ckpt = torch.load('./model/theta=90/ourModel.pt', map_location=device)
            model.load_state_dict(ckpt, strict=False)
            model.eval()
        elif low_res.shape[1] == 60:
            ckpt = torch.load('./model/theta=60/ourModel.pt', map_location=device)
            model.load_state_dict(ckpt, strict=False)
            model.eval()
        else:
            ckpt = torch.load('./model/theta=45/ourModel.pt', map_location=device)
            model.load_state_dict(ckpt, strict=False)
            model.eval()
        lr = np.array(low_res)
        lr_tensor = torch.Tensor(lr).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
        scale = args.sr_size[0] / lr_tensor.size(2)
        scale2 = args.sr_size[1] / lr_tensor.size(3)
        model.set_scale(scale, scale2)
        coord = make_coord((lr_tensor.size(2), lr_tensor.size(3))).unsqueeze(0).to(device)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / args.sr_size[0]
        cell[:, 1] *= 2 / args.sr_size[1]
        tensor_list = [lr_tensor, coord, cell]
        high_res = model(tensor_list)
        sr = utility.quantize(high_res, args.rgb_range)
        sr = sr.data.mul(255 / args.rgb_range)
        sr = sr[0, ...].permute(1, 2, 0).cpu().numpy()
        return sr

    def __baseModel(self, sino_name):

        """
        相当于是直接调用已经写好的模型和参数，直接加载模型
        与 __ourModel 类似，但调用 baseline 模型：
        1. 根据图像尺寸选择对应模型（theta=90/60/45）；
        2. 将图像转换为Tensor，设置放大比例后调用模型生成结果。
        """

        device = self.device
        # 在这里改成我们的模型
        model = BaseLine(args).to(device)
        low_res = cv2.imread(sino_name)
        if low_res.shape[1] == 90:
            ckpt = torch.load('./model/theta=90/baseModel.pt', map_location=device)
            model.load_state_dict(ckpt, strict=False)
            model.eval()
        elif low_res.shape[1] == 60:
            ckpt = torch.load('./model/theta=60/baseModel.pt', map_location=device)
            model.load_state_dict(ckpt, strict=False)
            model.eval()
        else:
            ckpt = torch.load('./model/theta=45/baseModel.pt', map_location=device)
            model.load_state_dict(ckpt, strict=False)
            model.eval()
        lr = np.array(low_res)
        lr_tensor = torch.Tensor(lr).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
        scale = args.sr_size[0] / lr_tensor.size(2)
        scale2 = args.sr_size[1] / lr_tensor.size(3)
        model.set_scale(scale, scale2)
        sr = model(lr_tensor)
        sr = utility.quantize(sr, args.rgb_range)
        sr = sr.data.mul(255 / args.rgb_range)
        sr = sr[0, ...].permute(1, 2, 0).cpu().numpy()
        return sr

    def __baseImage(self):

        """
        调用 baseline 模型处理图像：
        检查是否有正弦图像加载，搜索CUDA设备，如果内存充足则调用 __baseModel，
        将结果保存并显示，最后释放显存。
        """

        if self.radonView.items() == []:
            QMessageBox.information(self, 'Notice', 'Missing image！')
        else:
            self.__searchDevice()
            if self.deviceFlag == False:
                QMessageBox.warning(self, 'Warning', 'Out of memory！')
            else:
                QMessageBox.information(self, 'Notice', 'Using the model,please wait patiently！')
                with alive_bar(1) as bar:
                    SR_res = self.__baseModel(self.sinoName)
                    cv2.imwrite(self.baseSinoName, SR_res)
                    self.__loadImage(self.baseSinoName, window=3)
                    time.sleep(0.01)
                    bar()
                torch.cuda.empty_cache()
                QMessageBox.information(self, 'Notice', 'Super-resolution done！')

    def __srImage(self):
        """
        调用超分模型处理图像，与 __baseImage 类似，但调用 __ourModel。
        """
        if self.radonView.items() == []:
            QMessageBox.information(self, 'Notice', 'Missing image！')
        else:
            self.__searchDevice()
            if self.deviceFlag == False:
                QMessageBox.warning(self, 'Warning', 'Out of memory！')
            else:
                QMessageBox.information(self, 'Notice', 'Using the model,please wait patiently！')
                with alive_bar(1) as bar:
                    SR_res = self.__ourModel(self.sinoName)
                    cv2.imwrite(self.srSinoName, SR_res)
                    self.__loadImage(self.srSinoName, window=3)
                    time.sleep(0.01)
                    bar()
                torch.cuda.empty_cache()
                QMessageBox.information(self, 'Notice', 'Super-resolution done！')

    # -------------------------插值方法 ---------------------------
    def __nearestImage(self):
        """
        最近邻插值：
        读取正弦图图像，仅对 y 轴进行插值，
        弹出对话框让用户选择期望的新图像高度，
        插值后保存结果并加载显示。
        """
        if self.radonView.items() == []:
            QMessageBox.information(self, 'Notice', 'Missing image！')
        else:
            low_res = cv2.imread(self.sinoName, cv2.IMREAD_GRAYSCALE)

            # 弹出对话框，使用你的 angleWindow（可改名为尺寸选择窗口，但这里复用已有类）
            dialog = angleWindow(self)
            # 注意：exec_() 会阻塞窗口，等待用户确认选择
            if dialog.exec_() == QDialog.Accepted:
                # 获取用户选择的索引（假设 comboBox 中设置的选项代表新高度的对应项）
                index = dialog.comboBox.currentIndex()
                # 将选择的索引映射到具体的高度值
                size_mapping = {0: 180, 1:55, 2: 45, 3: 35}
                new_height = size_mapping.get(index, low_res.shape[0])
            else:
                # 如果用户取消，则使用原始高度
                new_height = low_res.shape[0]

            # 注意这里：由于只改变高度，所以传入的输出宽度保持 low_res.shape[1]
            INTER_res = cv2.resize(low_res, (low_res.shape[1], new_height), interpolation=cv2.INTER_NEAREST)

            # 保存并加载插值结果
            matplotlib.image.imsave(self.nearSinoName, INTER_res, cmap='gray')
            self.__loadImage(self.nearSinoName, window=3)
            QMessageBox.information(self, 'Notice', 'Nearest-neighbor interpolation done！')

    def __bilinearImage(self):
        """
        双线性插值：
        与 __nearestImage 类似，只是调用 cv2.INTER_LINEAR 插值。
        """
        if self.radonView.items() == []:
            QMessageBox.information(self, 'Notice', 'Missing image！')
        else:
            low_res = cv2.imread(self.sinoName, cv2.IMREAD_GRAYSCALE)

            # 弹出对话框，使用你的 angleWindow（可改名为尺寸选择窗口，但这里复用已有类）
            dialog = angleWindow(self)
            # 注意：exec_() 会阻塞窗口，等待用户确认选择
            if dialog.exec_() == QDialog.Accepted:
                # 获取用户选择的索引（假设 comboBox 中设置的选项代表新高度的对应项）
                index = dialog.comboBox.currentIndex()
                # 将选择的索引映射到具体的高度值
                size_mapping = {0: 180, 1: 60, 2: 45, 3: 35}
                new_height = size_mapping.get(index, low_res.shape[0])
            else:
                # 如果用户取消，则使用原始高度
                new_height = low_res.shape[0]

            # 注意这里：由于只改变高度，所以传入的输出宽度保持 low_res.shape[1]
            INTER_res = cv2.resize(low_res, (low_res.shape[1], new_height), interpolation=cv2.INTER_LINEAR)

            # 保存并加载插值结果
            matplotlib.image.imsave(self.nearSinoName, INTER_res, cmap='gray')
            self.__loadImage(self.nearSinoName, window=3)
            QMessageBox.information(self, 'Notice', 'Bilinear interpolation done！')

    def __bicubicImage(self):
        """
        双三次插值：
        调用 cv2.INTER_CUBIC 实现双三次插值，并保存显示结果。
        """
        if self.radonView.items() == []:
            QMessageBox.information(self, 'Notice', 'Missing image！')
        else:
            low_res = cv2.imread(self.sinoName, cv2.IMREAD_GRAYSCALE)

            # 弹出对话框，使用你的 angleWindow（可改名为尺寸选择窗口，但这里复用已有类）
            dialog = angleWindow(self)
            # 注意：exec_() 会阻塞窗口，等待用户确认选择
            if dialog.exec_() == QDialog.Accepted:
                # 获取用户选择的索引（假设 comboBox 中设置的选项代表新高度的对应项）
                index = dialog.comboBox.currentIndex()
                # 将选择的索引映射到具体的高度值
                size_mapping = {0: 180, 1:55, 2: 45, 3: 35}
                new_height = size_mapping.get(index, low_res.shape[0])
            else:
                # 如果用户取消，则使用原始高度
                new_height = low_res.shape[0]

            # 注意这里：由于只改变高度，所以传入的输出宽度保持 low_res.shape[1]
            INTER_res = cv2.resize(low_res, (low_res.shape[1], new_height), interpolation=cv2.INTER_CUBIC)

            # 保存并加载插值结果
            matplotlib.image.imsave(self.nearSinoName, INTER_res, cmap='gray')
            self.__loadImage(self.nearSinoName, window=3)
            QMessageBox.information(self, 'Notice', 'Bicubic interpolation done！')

    def calculate_psnr(self, gt, recon):
        # 计算PSNR（支持彩色图像）
        if gt.shape != recon.shape:
            raise ValueError("Input images must have the same dimensions")
        mse = np.mean((gt.astype(np.float32) - recon.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return psnr

    def interactive_ssim_analysis(self, gt_path, recon_path):
        # 读取图像
        gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
        recon = cv2.cvtColor(cv2.imread(recon_path), cv2.COLOR_BGR2RGB)

        # ================== SSIM分析部分 ==================
        # 计算全局SSIM和差异图
        ssim_value, ssim_map = structural_similarity(
            cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(recon, cv2.COLOR_RGB2GRAY),
            full=True
        )
        diff = ssim_map  # 确保差异值越大表示差异越大
        # 创建SSIM热力图（显示全局SSIM）
        fig_ssim = px.imshow(
            diff,
            color_continuous_scale='jet',
            labels={'color': 'SSIM Difference'},
            title=f'SSIM Difference Map (Global SSIM: {ssim_value:.4f})'
        )
        fig_ssim.update_traces(
            hovertemplate="<b>X</b>: %{x}<br><b>Y</b>: %{y}<br><b>Diff</b>: %{z:.2f}<extra></extra>"
        )

        # ================== PSNR分析部分 ==================
        # 计算PSNR
        psnr_value = self.calculate_psnr(gt, recon)

        # 创建3D误差表面图（显示PSNR）
        surface = go.Surface(z=diff, colorscale='jet')
        fig_3d = go.Figure(data=[surface])
        fig_3d.update_layout(
            title=f'3D Error Surface (PSNR: {psnr_value:.2f} dB)',
            scene=dict(zaxis_title='Error Magnitude')
        )

        # 使用plotly.io.show来强制弹出新窗口
        pio.show(fig_ssim)  # 显示SSIM分析结果的弹窗
        pio.show(fig_3d)  # 显示PSNR分析结果的弹窗

    # -------------------------归因图 ---------------------------
    def __lamImage(self):
        """
        调用归因分析：
        1. 检查是否有超分/插值结果图（srView）的图像；
        2. 如果当前加载的图像是通过超分或baseline处理（t_Fname 与 srSinoName 或 baseSinoName匹配），
           则打开归因分析窗口（lamWindow）；
        3. 否则提示不支持当前算法进行归因分析。
        """

        if not self.srView.items():
            QMessageBox.information(self, 'Notice', 'Missing image！')
        else:
            if self.t_Fname == self.srSinoName or self.t_Fname == self.baseSinoName:
                # 生成并显示归因分析窗口
                self.__lamWindow = PlotWindow(self.reconName, self.Fname)
                self.__lamWindow.raise_()
                self.__lamWindow.show()
            else:
                QMessageBox.warning(self, 'Warning',
                                    'The interpolation algorithm can not perform attribution analysis！')
        # if self.srView.items() == []:
        #     QMessageBox.information(self, 'Notice', 'Missing image！')
        # else:
        #     # 使用GT图像和SR图像进行SSIM与PSNR分析
        #     gt_image = self.Fname  # 假设这是GT图像路径
        #     recon_image = self.reconName  # 假设这是重建图像路径
        #
        #     # 调用SSIM和PSNR分析函数
        #     self.interactive_ssim_analysis(gt_image, recon_image)

    def reconFBP(self, proj, img_size):
        """
        使用 ODL 的 FBP 算法进行 CT 图像重建

        参数：
            proj (ndarray)：投影数据，形状为 (num_angles, detector_num)
            img_size (int)：重建图像的尺寸（默认 img_size x img_size）
        返回：
            reco (ndarray)：重建后的图像（NumPy 数组）
        """
        num_angles, detector_num = proj.shape

        # 构造角度采样：均匀分布在 [0, π)
        theta = np.linspace(0, np.pi, num_angles, endpoint=False)
        grid = odl.RectGrid(theta)
        angles = odl.uniform_partition_fromgrid(grid)

        # 定义检测器分割
        detector_partition = odl.uniform_partition(-1, 1, detector_num)

        # 构造射线几何模型（平行射线）
        geometry = odl.tomo.Parallel2dGeometry(angles, detector_partition)

        # 定义重建空间
        reco_space = odl.uniform_discr(
            min_pt=[-1, -1],
            max_pt=[1, 1],
            shape=(img_size, img_size),
            dtype='float32'
        )

        # 创建射线变换对象（这里尝试使用 astra_cuda，如有问题可替换为 'astra_cpu'）
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

        # 创建 FBP 重建算子，滤波器选择 'ram-lak'
        # fbp = odl.tomo.fbp_op(ray_trafo, filter_type='ram-lak')
        fbp = odl.tomo.fbp_op(ray_trafo)
        print(proj.max(),proj.min())
        # 执行重建
        reco = fbp(proj).asarray()
        print(f"reconImg.max: {reco.max()}\treconImg.min: {reco.min()}\n")
        return reco

    def __iradonImage(self):

        """
        FBP 重建：
        1. 检查是否有超分/插值结果图（srView）；
        2. 提示等待时间；
        3. 读取存储 sinogram 的图像（self.sinoName），利用 FBP 算法进行 CT 重建（调用 reconFBP），
           将重建结果归一化并保存为 'recon.png'，最后加载显示到窗口4。
        """

        if self.srView.items() == []:
            QMessageBox.warning(self, 'Warning', 'Missing image！')
        else:
            QMessageBox.information(self, 'Notice', 'It takes about 20 seconds, please wait patiently！')

            # 读取 sinogram 图像（假设 sinogram 存在 self.sinoName 路径下，已经通过其他流程生成）
            projs = cv2.imread(self.sinoName, cv2.IMREAD_GRAYSCALE)
            if projs is None:
                QMessageBox.warning(self, 'Warning', 'Failed to load sinogram image！')
                return

            # 获取原始图像作为参考，取长边作为重建图像的尺寸
            orig = cv2.imread(self.Fname, cv2.IMREAD_GRAYSCALE)
            if orig is None:
                QMessageBox.warning(self, 'Warning', 'Failed to load original image！')
                return
            img_size = orig.shape[0]  # 取高度和宽度中的较大者作为 img_size

            # 调用 FBP 重建函数，生成重建图像（reco 为浮点型数组）
            reco = self.reconFBP(projs, img_size)

            cv2.imwrite(self.reconName, reco)

            # 将生成的重建图加载到窗口4显示
            self.__loadImage(self.reconName, window=4)
            QMessageBox.information(self, 'Notice', 'FBP reconstruction done！')

    # # -------------------------反拉东变换 ---------------------------
    # def __iradonImage(self):
    #     """
    #     反拉东变换：
    #     1. 检查是否有超分/插值结果图（srView）的图像；
    #     2. 提示等待时间后，调用外部脚本（run_myIRadon.sh）进行反拉东变换，
    #        并将生成的重建图（recon.png）加载显示到窗口4。
    #     """
    #     if self.srView.items() == []:
    #         QMessageBox.warning(self, 'Warning', 'Missing image！')
    #     else:
    #         QMessageBox.information(self, 'Notice', 'It takes about 20 seconds, please wait patiently！')
    #         os.system(f'../myIRadon/for_testing/run_myIRadon.sh {self.t_Fname}')
    #         self.__loadImage('recon.png', window=4)
    #         QMessageBox.information(self, 'Notice', 'Iradon done！')

    # -------------------------一键处理按钮 ---------------------------
    def begin_clicked(self):
        """
        一键处理：
        1. 如果已经加载图片，则先根据角度控件执行拉东变换（angleChanged_v1）；
        2. 根据当前选中的单选按钮判断处理方式：插值、超分或归因分析，
           分别调用 handle_interpolation、__srImage 或 __lamImage；
        3. 最后执行反拉东变换。
        """
        if self.Fname:
            angle = self.angle.value()
            self.angleChanged_v1()
        else:
            QMessageBox.information(self, 'Notice', 'Missing image！')

        if self.Interpolate1.isChecked():
            interpolate_method = self.Interpolate2.currentText()
            self.handle_interpolation(interpolate_method, angle)
        elif self.ArbRCAN.isChecked():
            angle = self.angle.value()
            self.__srImage(angle)
        elif self.LAM.isChecked():
            angle = self.angle.value()
            self.__lamImage(angle)
        else:
            QMessageBox.warning(self, "Warning", "Please select a processing method")
            return
        self.__iradonImage()

    def handle_interpolation(self, method, angle):
        """
        根据用户选择的插值方法（下拉框选项）调用对应函数：
        Bilinear、Bicubic 或 Nearest-neighbor 插值。

        """
        if method == "Bilinear":
            self.__bilinearImage(angle)
        elif method == "Bicubic":
            self.__bicubicImage(angle)
        elif method == "Nearest-neighbor":
            self.__nearestImage(angle)
        else:
            QMessageBox.warning(self, "Warning", "Invalid interpolation method")

    def handleAngle(self, index):

        """
        根据 index 获取对应角度值，并调用相应的拉东变换脚本。
        此函数映射 index 到角度值，并返回脚本执行是否成功的布尔值。
        """
        angle_index_map = {0: 180, 1: 90, 2: 55, 3: 45}
        if index not in angle_index_map:
            QMessageBox.warning(self, "Index Error", "Invalid index value")
            return False
        angle = angle_index_map[index]
        self.radonAngle = angle
        self.sinoName = self.tempDir + f'sino_{angle}.png'
        angle_dir_map = {180: 'myRadon', 90: 'myRadon_90', 60: 'myRadon_60', 45: 'myRadon_45'}
        dir_name = angle_dir_map[angle]
        project_root = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(project_root, dir_name, 'for_testing', 'run_myRadon.sh')
        flag1 = os.system(f'"{script_path}" {self.Fname} {self.sinoName}')
        return flag1 == 0


########################################################################
# 投影角度选择子窗口（对话框）
########################################################################

class angleWindow(QDialog, Ui_angleChoice):

    """
    角度选择对话框：
    1. 用户通过下拉框选择拉东变换的角度；
    2. 点击确定后通过自定义信号 index 发出当前选项的索引，
       主窗口收到信号后调用 angleChanged 方法进行处理。
    """

    index = pyqtSignal(int)

    def __init__(self, parent=None):
        super(angleWindow, self).__init__(parent)
        self.setupUi(self)
        self.__center()
        self.buttonBox.accepted.connect(self.selectionChange)
        self.buttonBox.rejected.connect(self.exit)

    def selectionChange(self):
        self.index.emit(self.comboBox.currentIndex())
        self.close()

    def exit(self):
        self.close()

    def __center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


########################################################################
# 加载动画窗口（用于归因图计算时显示等待动画）
########################################################################

class Loading_Win(QWidget):
    """
    加载动画窗口：
    1. 无边框、对话框、置顶显示；
    2. 使用 QMovie 播放 gif 动画；
    3. 设置窗口模态，阻止用户操作主窗口；
    4. 当 lamFlag 为 True 时直接关闭窗口（用于归因图计算完毕后）。
    """

    def __init__(self, flag):
        super().__init__()
        self.resize(120, 120)
        self.__center()
        self.lamFlag = flag
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.loading_gif = QMovie('qt/icon/load.gif')
        self.loading_label = QLabel(self)
        self.loading_label.setMovie(self.loading_gif)
        self.loading_gif.start()
        self.setWindowModality(Qt.ApplicationModal)
        if not self.lamFlag:
            self.show()
        else:
            self.close()

    def __center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


########################################################################
# 重建结果归因分析展示界面
########################################################################

class PlotAnalysis:
    def __init__(self, gt_path, recon_path):
        self.gt_path = gt_path
        self.recon_path = recon_path

    def calculate_psnr(self, gt, recon):
        mse = np.mean((gt.astype(float) - recon.astype(float)) ** 2)
        return 10 * np.log10(255.0 ** 2 / mse) if mse != 0 else float('inf')

    def generate_ssim_plot(self):
        """生成SSIM热力图"""
        gt = cv2.cvtColor(cv2.imread(self.gt_path), cv2.COLOR_BGR2RGB)
        recon = cv2.cvtColor(cv2.imread(self.recon_path), cv2.COLOR_BGR2RGB)

        # 计算SSIM
        ssim_value, ssim_map = structural_similarity(
            cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(recon, cv2.COLOR_RGB2GRAY),
            full=True
        )
        diff = ssim_map

        psnr_value = self.calculate_psnr(gt, recon)
        # 生成热力图
        fig = px.imshow(
            diff,
            color_continuous_scale='jet',
            labels={'color': 'SSIM 差异'},
            title=f'SSIM 差异热力图'
                  f'(全局 SSIM: {ssim_value:.4f})'
                  f'(全局 PSNR: {psnr_value:.2f} dB)'
        )

        # 保存为临时文件
        temp_dir = tempfile.mkdtemp()
        html_path = os.path.join(temp_dir, "ssim_plot.html")
        fig.write_html(html_path, include_plotlyjs='cdn')
        return html_path

    # def generate_psnr_3d_plot(self):
    #     """生成PSNR 3D图"""
    #     gt = cv2.cvtColor(cv2.imread(self.gt_path), cv2.COLOR_BGR2RGB)
    #     recon = cv2.cvtColor(cv2.imread(self.recon_path), cv2.COLOR_BGR2RGB)
    #
    #     # 计算PSNR和差异
    #     psnr_value = self.calculate_psnr(gt, recon)
    #     _, ssim_map = structural_similarity(
    #         cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY),
    #         cv2.cvtColor(recon, cv2.COLOR_RGB2GRAY),
    #         full=True
    #     )
    #     diff = 1 - ssim_map  # 使用SSIM差异作为示例数据
    #
    #     # 生成3D图
    #     fig = go.Figure(data=[go.Surface(z=diff, colorscale='jet')])
    #     fig.update_layout(
    #         title=f'3D 误差表面 (PSNR: {psnr_value:.2f} dB)',
    #         scene=dict(zaxis_title='误差强度')
    #     )
    #
    #     # 保存为临时文件
    #     temp_dir = tempfile.mkdtemp()
    #     html_path = os.path.join(temp_dir, "psnr_3d_plot.html")
    #     fig.write_html(html_path, include_plotlyjs='cdn')
    #     return html_path


class PlotWindow(QMainWindow):
    def __init__(self, html_path):
        super().__init__()
        self.setWindowTitle("分析结果")
        self.setGeometry(100, 100, 1200, 800)
        self.__center()

        # 创建浏览器组件并加载HTML
        self.browser = QWebEngineView()
        self.browser.load(QUrl.fromLocalFile(html_path))

        # 设置布局
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.browser)
        self.setCentralWidget(central_widget)

    def __center(self):
        """窗口居中"""
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


########################################################################
# 自定义 QLabel 类，用于实现鼠标拖拽选择区域
########################################################################

class MyLabel(QLabel):
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    flag = False

    def mousePressEvent(self, event):
        """
        鼠标按下事件：
        记录起始坐标，并设置标志 flag 为 True。
        """
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()

    def mouseReleaseEvent(self, event):
        """
        鼠标释放事件：
        将标志 flag 设为 False。
        """
        self.flag = False

    def mouseMoveEvent(self, event):
        """
        鼠标移动事件：
        如果鼠标按下状态，记录当前位置并调用 update 重绘区域选择矩形。
        """
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()

    def paintEvent(self, event):
        """
        重写绘制事件：
        在图像上绘制一个红色矩形，表示用户选择的区域。
        若宽高相等则绘制正方形，否则按水平宽度绘制正方形。
        """
        super().paintEvent(event)
        if abs(self.x1 - self.x0) == abs(self.y1 - self.y0):
            rect = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(rect)
        else:
            rect = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.x1 - self.x0))
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(rect)


########################################################################
# 鼠标事件追踪器
########################################################################

class MouseTracker(QtCore.QObject):
    """
    通过事件过滤器追踪指定 widget 的鼠标点击和释放事件，
    并通过自定义信号 positionChanged 和 windowChanged 将鼠标坐标传递出去。
    """
    positionChanged = QtCore.pyqtSignal(QtCore.QPoint)
    windowChanged = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, widget):
        super().__init__(widget)
        self._widget = widget
        self.widget.setMouseTracking(True)
        self.widget.installEventFilter(self)

    @property
    def widget(self):
        return self._widget

    def eventFilter(self, o, e):
        if o is self.widget and e.type() == QtCore.QEvent.MouseButtonPress:
            self.positionChanged.emit(e.pos())
        elif o is self.widget and e.type() == QtCore.QEvent.MouseButtonRelease:
            self.windowChanged.emit(e.pos())
        return super().eventFilter(o, e)


########################################################################
# 保存选择对话框
########################################################################

class saveWindow(QDialog, Ui_saveChoice):
    """
    保存选择对话框：
    1. 用户通过下拉框选择要保存的图像类型；
    2. 点击确定后通过信号 index 发出选择的索引。
    """
    index = pyqtSignal(int)

    def __init__(self):
        super(saveWindow, self).__init__()
        self.setupUi(self)
        self.__center()
        self.buttonBox.accepted.connect(self.saveImage)
        self.buttonBox.rejected.connect(self.cancelSave)

    def saveImage(self):
        self.index.emit(self.comboBox.currentIndex())
        self.close()

    def cancelSave(self):
        self.close()

    def __center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


########################################################################
# 程序入口
########################################################################


if __name__ == '__main__':
    """
    程序入口：
    1. 创建 QApplication 实例，并加载 qdarkstyle 样式；
    2. 创建并显示登录窗口；
    3. 创建主窗口实例（登录成功后将调用 mainWindow.show()）。
    """

    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    loginWindow = loginWindow()
    loginWindow.show()
    mainWindow = MainWindow()
    sys.exit(app.exec_())
