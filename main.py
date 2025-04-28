import sys
from threading import Thread
# 从 qt 模块下导入各个UI文件（由 Qt Designer 生成的类），这些类用于构造各个界面
from qt.sino_res import Ui_RES_SINO
from PyQt5.QtCore import Qt, QRect, pyqtSignal
from alive_progress import alive_bar
import time
from option import args  # 超分模型、baseline等所需的参数配置
from arbrcan import ArbRCAN  # 超分模型类（使用 ArbRCAN 网络）
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
    QDesktopWidget, QDialog
from PyQt5.QtWidgets import QLabel, QWidget, QApplication, QMainWindow
import matplotlib.pyplot as plt
from qt.mainWindow_v11 import Ui_MainWindow  # 主窗口的UI
from qt.login import Ui_Login  # 登录窗口的UI
from qt.lam_v3 import Ui_LAM  # 归因分析界面UI
import numpy as np
import torch
from LAM.ModelZoo.utils import PIL2Tensor  # 将PIL图像转换为Tensor的工具函数
from LAM.ModelZoo import load_model  # 加载预训练模型的函数
# 以下导入归因分析相关的工具函数和方法
from LAM.SaliencyModel.utils import vis_saliency, vis_saliency_kde, grad_abs_norm, prepare_images
from LAM.SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from LAM.SaliencyModel.attributes import attr_grad
from LAM.SaliencyModel.BackProp import attribution_objective, Path_gradient, Path_gradient1
from LAM.SaliencyModel.BackProp import saliency_map_PG as saliency_map
from LAM.SaliencyModel.BackProp import GaussianBlurPath
import cv2
from qt.lam_result_v3 import Ui_LAM_RESULT  # 归因结果展示界面UI
from qt.angle import Ui_angleChoice  # 投影角度选择对话框UI
from qt.save import Ui_saveChoice  # 保存选择对话框UI
import qdarkstyle  # 黑暗风格的样式
import pynvml  # NVIDIA显卡内存管理工具
from qt.recon_res import Ui_RES_RECON  # CT图残差窗口UI
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import sqlite3


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
            mainWindow.showFullScreen()  # 可全屏显示
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
        self.setupUi(self)  # 根据 Ui_MainWindow 构造UI
        self.__center()  # 居中显示
        self.deviceFlag = False  # 标志是否找到了可用的CUDA设备
        self.__lamWidow = None  # 用于归因图窗口的实例
        self.Fname = None  # 当前打开的图片文件名
        self.t_Fname = None  # 临时图片文件名
        # 初始化归因图参数及角度设置
        # x和y表示某个点图像区域的坐标
        self.x = 0
        self.y = 0
        self.wis = 0
        # 默认拉东变换的坐标是180度
        self.radonAngle = 180
        # 设置图片存放路径，用于加载图片进行处理
        self.gtDir = './ct_gt/'  # 原图目录
        self.tempDir = './temp/'  # 中间文件目录
        self.saveDir = './save/'  #保存目录
        # 定义具体的文件名
        self.gtName = 'GT.png'
        # 超分后得到的sinogram
        # 修改成员变量命名（可选）
        self.srSinoName = self.tempDir + 'recon_result.png'


        # self.srSinoName = self.tempDir + 'srSino.png'


        # baseline模型处理后得到的正弦图
        self.baseSinoName = self.tempDir + 'baseSino.png'
        # nearSinoName、bilSinoName、bicSinoName 分别用于保存最近邻、双线性、双三次插值后的图像。
        self.nearSinoName = self.tempDir + 'nearSino.png'
        self.bilSinoName = self.tempDir + 'bilSino.png'
        self.bicSinoName = self.tempDir + 'bicSino.png'
        self.angle.setValue(180)  # 初始化角度控件为180°

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
        # self.actionModel.triggered.connect(self.__srImage)
        # # baseline模型菜单
        # self.actionBaseline.triggered.connect(self.__baseImage)
        # 三种插值算法菜单
        self.actionNearest_neighbor.triggered.connect(self.__nearestImage)
        self.actionBilinear.triggered.connect(self.__bilinearImage)
        self.actionBicubic.triggered.connect(self.__bicubicImage)
        # 归因图菜单，绑定归因图处理，调用_lamImage方法
        self.actionLAM.triggered.connect(self.__lamImage)
        # 反拉东变换菜单
        self.actionIradon.triggered.connect(self.__iradonImage)
        # 查看残差图功能
        self.actionsinoDiff.triggered.connect(self.__sinoDiff)
        self.actionreconDiff.triggered.connect(self.__reconDiff)
        # 一键处理按钮绑定
        self.begin.clicked.connect(self.begin_clicked)

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


    # -------------------------CT图残差（重建）窗口 ---------------------------

    def __reconDiff(self):
        """
        当需要比较原始CT图与重建图的差异时：
        1. 检查反拉东变换显示区域是否存在图像；
        2. 根据当前输入信息构造文件名，并调用 res_reconWindow 显示残差图。
        """
        if self.iradonView.items() == []:
            QMessageBox.information(self, 'Notice', 'Missing image！')
        else:
            gt_name = self.gtDir + self.src_name.text() + '.png'
            recon_name = self.recon_name.text() + '.png'
            self.reconDiffWin = res_reconWindow(gt_name, recon_name)
            self.reconDiffWin.show()

    # -------------------------正弦图残差窗口 ---------------------------
    def __sinoDiff(self):

        """
        显示正弦图残差：
        如果相关图像存在，则生成残差图（使用 cv2.subtract 计算差异），
        并调用 res_sinoWindow 展示对比结果（SSIM、PSNR等指标也会显示）。
        """

        if self.srView.items() == []:
            QMessageBox.information(self, 'Notice', 'Missing image！')
        else:
            self.sinoDiffWin = res_sinoWindow(self.gtName, self.t_Fname)
            self.sinoDiffWin.show()

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

    # -------------------------打开文件 ---------------------------
    def __openFileAndShowImage(self):
        """
        弹出文件选择对话框，选择图片后调用 __loadImage 加载，并在界面显示原图。
        """
        fname, _ = QFileDialog.getOpenFileName(self, 'Open File', self.gtDir,
                                               "Picture Files(*.png *.jpg *.jpeg *.gif *.bmp)")
        if fname:
            self.Fname = fname
            self.__loadImage(fname, window=1)
        else:
            QMessageBox.information(self, 'Notice', 'No file selected！')

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
                current_img = cv2.imread(self.gtDir + self.src_name.text() + '.png')
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
                    current_img = cv2.imread(self.gtDir + self.src_name.text() + '.png')
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
            self.angleWinow.index.connect(self.angleChanged)
        else:
            QMessageBox.information(self, 'Notice', 'Missing image！')

    def radon_GT(self):
        """
        使用系统命令调用外部脚本生成 GT 图像。
        如果需要替换拉东变换算法，可以修改这里调用的脚本或直接内嵌实现代码。
        """
        os.system(f'./myRadon/for_testing/run_myRadon.sh {self.Fname} {self.gtName}')

    def radon_image(self, index):
        """
        根据选择的角度（index）调用相应的拉东变换脚本，并生成对应的 sinogram 图片。
        index=0：180°、1：90°、2：60°、3：45°。调用外部脚本生成图片后返回 True 表示成功。
        """
        flag1 = 1
        if index == 0:
            self.radonAngle = 180
            self.sinoName = self.tempDir + 'sino_{}.png'.format(self.radonAngle)
            flag1 = os.system(f'./myRadon/for_testing/run_myRadon.sh {self.Fname} {self.sinoName}')
        elif index == 1:
            self.radonAngle = 90
            self.sinoName = self.tempDir + 'sino_{}.png'.format(self.radonAngle)
            flag1 = os.system(f'./myRadon_90/for_testing/run_myRadon.sh {self.Fname} {self.sinoName}')
        elif index == 2:
            self.radonAngle = 60
            self.sinoName = self.tempDir + 'sino_{}.png'.format(self.radonAngle)
            flag1 = os.system(f'./myRadon_60/for_testing/run_myRadon.sh {self.Fname} {self.sinoName}')
        elif index == 3:
            self.radonAngle = 45
            self.sinoName = self.tempDir + 'sino_{}.png'.format(self.radonAngle)
            flag1 = os.system(f'./myRadon_45/for_testing/run_myRadon.sh {self.Fname} {self.sinoName}')
        if flag1 == 0:
            return True

    def angleChanged(self, index):
        """
        投影角度选择对话框返回后：
        1. 弹出提示等待时间；
        2. 利用线程池并行调用生成 GT 图和拉东图；
        3. 当任务完成后，加载并显示生成的 sinogram 图像到窗口2。
        """
        QMessageBox.information(self, 'Notice', 'It will take about 10 seconds, please be patient！')
        pool = ThreadPoolExecutor(max_workers=2)
        pool.submit(self.radon_GT)
        future2 = pool.submit(self.radon_image(index))
        if future2.done():
            self.__loadImage(self.sinoName, window=2)

    def angleChanged_v1(self):
        """
        一键处理时，依据角度控件的值执行拉东变换：
        将角度映射到对应的脚本索引，并并行调用生成GT图和拉东图，最后加载结果。
        """
        QMessageBox.information(self, 'Notice', 'It will take about 10 seconds, please be patient！')
        pool = ThreadPoolExecutor(max_workers=2)
        pool.submit(self.radon_GT)
        angle_index_map = {180: 0, 90: 1, 60: 2, 45: 3}
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
        读取正弦图图像，计算水平与垂直缩放因子，并使用 cv2.resize（INTER_NEAREST）进行插值，
        保存结果后加载显示。
        """
        if self.radonView.items() == []:
            QMessageBox.information(self, 'Notice', 'Missing image！')
        else:
            low_res = cv2.imread(self.sinoName, cv2.IMREAD_GRAYSCALE)
            fy = args.sr_size[0] / low_res.shape[0]
            fx = args.sr_size[1] / low_res.shape[1]
            INTER_res = cv2.resize(low_res, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
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
            fy = args.sr_size[0] / low_res.shape[0]
            fx = args.sr_size[1] / low_res.shape[1]
            INTER_res = cv2.resize(low_res, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
            matplotlib.image.imsave(self.bilSinoName, INTER_res, cmap='gray')
            self.__loadImage(self.bilSinoName, window=3)
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
            fy = args.sr_size[0] / low_res.shape[0]
            fx = args.sr_size[1] / low_res.shape[1]
            INTER_res = cv2.resize(low_res, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
            matplotlib.image.imsave(self.bicSinoName, INTER_res, cmap='gray')
            self.__loadImage(self.bicSinoName, window=3)
            QMessageBox.information(self, 'Notice', 'Bicubic interpolation done！')

    # -------------------------归因图 ---------------------------
    def __lamImage(self):
        """
        调用归因分析：
        1. 检查是否有超分/插值结果图（srView）的图像；
        2. 如果当前加载的图像是通过超分或baseline处理（t_Fname 与 srSinoName 或 baseSinoName匹配），
           则打开归因分析窗口（lamWindow）；
        3. 否则提示不支持当前算法进行归因分析。
        """
        if self.srView.items() == []:
            QMessageBox.information(self, 'Notice', 'Missing image！')
        else:
            if self.t_Fname == self.srSinoName or self.t_Fname == self.baseSinoName:
                self.__lamWidow = lamWindow(self.sinoName, self.gtName, self.t_Fname)
                self.__lamWidow.show()
            else:
                QMessageBox.warning(self, 'Warning',
                                    'The interpolation algorithm can not perform attribution analysis！')

    # -------------------------反拉东变换 ---------------------------
    def __iradonImage(self):
        """
        反拉东变换：
        1. 检查是否有超分/插值结果图（srView）的图像；
        2. 提示等待时间后，调用外部脚本（run_myIRadon.sh）进行反拉东变换，
           并将生成的重建图（recon.png）加载显示到窗口4。
        """
        if self.srView.items() == []:
            QMessageBox.warning(self, 'Warning', 'Missing image！')
        else:
            QMessageBox.information(self, 'Notice', 'It takes about 20 seconds, please wait patiently！')
            os.system(f'../myIRadon/for_testing/run_myIRadon.sh {self.t_Fname}')
            self.__loadImage('recon.png', window=4)
            QMessageBox.information(self, 'Notice', 'Iradon done！')

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
        angle_index_map = {0: 180, 1: 90, 2: 60, 3: 45}
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
# 归因结果展示界面
########################################################################

class lamRWindow(QWidget, Ui_LAM_RESULT):
    """
    归因结果窗口：
    1. 显示经过归因分析后生成的图像（区域标记和归因热图）；
    2. 显示 DI（Diffusion Index，扩散指数）值。
    """

    def __init__(self, x):
        super().__init__()
        self.setupUi(self)
        self.__center()
        self.Draw_img.setPixmap(QPixmap('./LAM/dram_img.png'))
        self.Lam_info.setPixmap(QPixmap('./LAM/informative_area.png'))
        self.label.setText('The DI of this case is %f' % x)

    def __center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


########################################################################
# 归因图界面
########################################################################

class lamWindow(QWidget, Ui_LAM):
    """
    归因图窗口：
    1. 显示原图（通过自定义的 MyLabel 实现鼠标拖拽选择区域）；
    2. 通过 MouseTracker 跟踪鼠标事件，得到选择区域的坐标和大小；
    3. 用户点击确认后，调用 LAMP 方法进行归因计算，该方法根据所选区域对归因模型进行计算；
    4. 归因结果计算完成后，调用 lamRWindow 展示结果。

    如果要替换归因分析方法，可以修改 LAMP 函数中调用的模型、路径函数或后处理算法。
    """
    mySignal = pyqtSignal(list)
    mySignal1 = pyqtSignal(bool)

    def __init__(self, lr_name, gt_name, sr_name):
        super().__init__()
        self.setupUi(self)
        self.__center()
        self.deviceFlag = False
        self.lamFlag = False
        self.list = []  # 用于存储选择区域的坐标及宽度
        self.DI = 0  # 扩散指数
        self.setWindowModality(Qt.ApplicationModal)
        self.imag_label = MyLabel(self)  # 自定义 Label 用于图像显示和区域选择
        self.imag_label.setGeometry(QRect(30, 30, 180, 515))
        self.imag_label.setPixmap(QPixmap(gt_name))
        self.imag_label.setCursor(Qt.CrossCursor)
        self.lr = lr_name
        self.sr = sr_name
        self.gt = gt_name
        tracker = MouseTracker(self.imag_label)
        tracker.positionChanged.connect(self.on_positionChanged)
        tracker.windowChanged.connect(self.on_windowChanged)
        self.confirmButton.clicked.connect(self.__confirmChanged)

    def __center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def on_positionChanged(self, pos):
        """
        记录鼠标按下时的坐标，用于确定选区起点
        """
        self.x = pos.x()
        self.y = pos.y()
        self.coord_label.setText("The coordinates of the starting point:({},{})".format(self.x, self.y))

    def on_windowChanged(self, pos):
        """
        记录鼠标释放时的坐标，根据起点和释放点计算选区的宽度
        """
        self.wis = abs(self.x - pos.x())
        self.list = [self.x, self.y, self.wis]
        self.windowsize_label.setText("The selected area size:({}*{})".format(self.wis, self.wis))

    def __confirmChanged(self):
        """
        当用户点击确认按钮后：
        1. 检查CUDA设备；
        2. 显示加载动画；
        3. 启动新的线程调用 LAMP 方法进行归因计算。
        """
        self.__searchDevice()
        if self.deviceFlag == False:
            QMessageBox.warning(self, 'Warning', 'Out of memory！')
        else:
            QMessageBox.information(self, 'Notice', 'Analyzing (takes about a minute), please wait patiently!')
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            self.loadWin = Loading_Win(self.lamFlag)
            self.close()
            self.t1 = Thread(target=self.LAMP,
                             args=(self.lr, self.gt, self.sr, self.list[2], self.list[0], self.list[1]))
            self.t1.start()

    def __searchDevice(self):
        """
        查找可用的CUDA设备，要求至少有19000MB空闲内存，类似于主窗口中的方法。
        """
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        for id in range(deviceCount):
            handler = pynvml.nvmlDeviceGetHandleByIndex(id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
            free = round(meminfo.free / 1024 / 1024, 2)
            if free >= 19000:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(id)
                self.device = 'cuda'
                self.deviceFlag = True

    def LAMP(self, lr_path, hr_path, sr_path, wis, x, y):
        """
        归因分析主流程：
        1. 根据 sr_path 的名称判断调用不同的模型（Base 或 RNAN）；
        2. 对输入图像进行预处理，将 PIL 图像转换为Tensor；
        3. 绘制选中区域在原图上的位置并保存（用于展示）；
        4. 根据归因目标、使用高斯模糊路径等方法计算梯度、归因图；
        5. 生成归因热图以及计算 Gini 指数，得到扩散指数 DI；
        6. 归因计算完成后关闭加载窗口，并启动 lamRWindow 显示结果。

        如果要替换归因方法，建议修改此函数中对模型调用、梯度计算和后处理部分。
        """
        device = self.device
        sigma = 1.2
        fold = 50
        l = 9
        alpha = 0.5
        window_size = wis
        x0 = x
        y0 = y
        name = sr_path.split('/')[-1]
        print(name)
        if name == 'baseSino.png':
            img_lr = prepare_images(lr_path)
            img_hr = prepare_images(hr_path)
            tensor_lr = PIL2Tensor(img_lr)[:3]
            scale = int(args.sr_size[1] / tensor_lr.shape[2])
            print('super scale:{}X1'.format(scale))
            model = load_model('SAN@Base', scale)
            draw_img = pil_to_cv2(img_hr)
            cv2.rectangle(draw_img, (x0, y0), (x0 + window_size, y0 + window_size), (0, 0, 255), 1)
            position_pil = cv2_to_pil(draw_img)
            position_pil.save('./LAM/dram_img.png')
            attr_objective = attribution_objective(attr_grad, y0, x0, window=window_size)
            gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
            interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient1(
                tensor_lr.numpy(), model, attr_objective, gaus_blur_path_func, cuda=True)
            grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
            abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
            saliency_image_abs = vis_saliency(abs_normed_grad_numpy, scale)
            saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy, scale)
            blend_kde_and_input = cv2_to_pil(
                pil_to_cv2(saliency_image_kde) * (1.0 - alpha) +
                pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
            saliency_image_abs.save('./LAM/lam_abs.png')
            blend_kde_and_input.save('./LAM/informative_area.png')
            gini_index = gini(abs_normed_grad_numpy)
            diffusion_index = (1 - gini_index) * 100
            self.lamFlag = True
            if self.lamFlag:
                torch.cuda.empty_cache()
                self.DI = diffusion_index
                self.loadWin.close()
                self.loadRWin = lamRWindow(self.DI)
                self.t2 = Thread(target=self.LAMR)
                self.t2.start()
        else:
            img_lr = prepare_images(lr_path)
            img_hr = prepare_images(hr_path)
            tensor_lr = PIL2Tensor(img_lr)[:3]
            scale = int(args.sr_size[1] / tensor_lr.shape[2])
            print('super scale:{}X1'.format(scale))
            model = load_model('RNAN@Base', scale)
            coord = make_coord((tensor_lr.shape[1], tensor_lr.shape[2])).unsqueeze(0).cuda()
            cell = torch.ones_like(coord)
            cell[:, 0] *= 2 / args.sr_size[0]
            cell[:, 1] *= 2 / args.sr_size[1]
            coord = coord.to(device)
            cell = cell.to(device)
            tensor_list = [tensor_lr, coord, cell]
            draw_img = pil_to_cv2(img_hr)
            cv2.rectangle(draw_img, (x0, y0), (x0 + window_size, y0 + window_size), (0, 0, 255), 1)
            position_pil = cv2_to_pil(draw_img)
            position_pil.save('./LAM/dram_img.png')
            attr_objective = attribution_objective(attr_grad, y0, x0, window=window_size)
            gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
            interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(
                tensor_list[0].numpy(), tensor_list[1], tensor_list[2],
                model, attr_objective, gaus_blur_path_func, cuda=True)
            grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
            abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
            saliency_image_abs = vis_saliency(abs_normed_grad_numpy, scale)
            saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy, scale)
            blend_kde_and_input = cv2_to_pil(
                pil_to_cv2(saliency_image_kde) * (1.0 - alpha) +
                pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
            saliency_image_abs.save('./LAM/lam_abs.png')
            blend_kde_and_input.save('./LAM/informative_area.png')
            gini_index = gini(abs_normed_grad_numpy)
            diffusion_index = (1 - gini_index) * 100
            self.lamFlag = True
            if self.lamFlag:
                torch.cuda.empty_cache()
                self.DI = diffusion_index
                self.loadWin.close()
                self.loadRWin = lamRWindow(self.DI)
                self.t2 = Thread(target=self.LAMR)
                self.t2.start()

    def LAMR(self):
        """
        归因结果窗口显示：
        当归因计算完成后，调用 lamRWindow 展示归因结果。
        """
        self.loadRWin.show()


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
# 正弦图残差窗口
########################################################################
class res_sinoWindow(QWidget, Ui_RES_SINO):
    """
    正弦图残差窗口：
    1. 读取GT图和超分/插值图（sr）并转为灰度图；
    2. 计算两图的差异（使用 cv2.subtract）；
    3. 保存并显示残差图，同时计算并显示 SSIM 和 PSNR。
    """

    def __init__(self, gtname, srname):
        super(res_sinoWindow, self).__init__()
        self.setupUi(self)
        self.__center()
        gt = cv2.imread(gtname, cv2.IMREAD_GRAYSCALE).astype(np.int32)
        sr = cv2.imread(srname, cv2.IMREAD_GRAYSCALE).astype(np.int32)
        res = cv2.subtract(gt, sr)
        plt.imsave('./temp/resSino.png', res, cmap='gray')
        self.gt_label.setPixmap(QPixmap(gtname))
        self.sr_label.setPixmap(QPixmap(srname))
        self.res_label.setPixmap(QPixmap('./temp/resSino.png'))
        result1 = np.zeros(gt.shape, dtype=np.float32)
        result2 = np.zeros(sr.shape, dtype=np.float32)
        cv2.normalize(gt, result1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.normalize(sr, result2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        ssim = compare_ssim(result1, result2)
        psnr = compare_psnr(result1, result2)
        self.ssim_label.setText(str(ssim))
        self.psnr_label.setText(str(psnr))

    def __center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


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
# CT图残差窗口
########################################################################

class res_reconWindow(QWidget, Ui_RES_RECON):
    """
    CT图残差窗口：
    1. 读取GT和重建图，计算两图差异；
    2. 保存残差图，并计算 SSIM、PSNR 指标后在窗口显示。
    """

    def __init__(self, GT_NAME, RECON_NAME):
        super(res_reconWindow, self).__init__()
        self.setupUi(self)
        self.__center()
        self.gt_label.setPixmap(QPixmap(GT_NAME))
        self.sr_label.setPixmap(QPixmap(RECON_NAME))
        gt = cv2.imread(GT_NAME, 0).astype(np.int32)
        sr = cv2.imread(RECON_NAME, 0).astype(np.int32)
        res = cv2.subtract(gt, sr)
        result1 = np.zeros(gt.shape, dtype=np.float32)
        result2 = np.zeros(sr.shape, dtype=np.float32)
        cv2.normalize(gt, result1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.normalize(sr, result2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        ssim = compare_ssim(result1, result2)
        psnr = compare_psnr(result1, result2)
        plt.imsave('./temp/resRecon.png', res, cmap='gray')
        self.res_label.setPixmap(QPixmap('./temp/resRecon.png'))
        self.ssim_label.setText(str(ssim))
        self.psnr_label.setText(str(psnr))

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
