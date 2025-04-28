# gui/main_window.py
import sys, os
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QHBoxLayout, QVBoxLayout, QListWidget,
    QListWidgetItem, QLabel, QPushButton, QMessageBox, QFrame, QFileDialog, QAction, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("平行锥束CT模拟软件")
        self.resize(1200, 800)

        # 固定工作区路径
        self.working_folder = r"E:\备份\exp-soft\imagep1\ct_gt"
        self.selected_image_path = None

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # 左侧区域
        left_frame = QFrame()
        left_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)
        main_layout.addWidget(left_frame, 1)  # 左侧占 1 份

        # 顶部导航
        header_label = QLabel("工作区")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        left_layout.addWidget(header_label)

        # 文件列表区域，缩小宽度
        self.list_widget = QListWidget()
        self.list_widget.setFixedWidth(150)  # 调整为更小宽度
        self.populateWorkingArea()
        self.list_widget.itemClicked.connect(self.onWorkingItemClicked)
        left_layout.addWidget(self.list_widget)

        # 右侧区域
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)
        main_layout.addWidget(right_frame, 0)  # 右侧占 3 份，平衡宽度

        # 顶部区域：图像预览和确认按钮
        preview_frame = QFrame()
        preview_layout = QHBoxLayout(preview_frame)
        preview_layout.setContentsMargins(5, 5, 5, 5)
        preview_layout.setSpacing(10)

        self.preview_label = QLabel("图像预览")
        self.preview_label.setFixedSize(400, 400)  # 正方形预览区域
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid black;")
        preview_layout.addWidget(self.preview_label)

        self.confirm_button = QPushButton("确定选择该图像作为实验图像")
        self.confirm_button.setFixedWidth(200)
        self.confirm_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.confirm_button.clicked.connect(self.onConfirmSelection)
        preview_layout.addWidget(self.confirm_button)

        right_layout.addWidget(preview_frame)

        # 下部区域：原有功能占位区（如 Radon 变换、CT 重建、差异分析等）
        placeholder_frame = QFrame()
        placeholder_layout = QVBoxLayout(placeholder_frame)
        placeholder_layout.setContentsMargins(5, 5, 5, 5)
        placeholder_layout.setSpacing(5)
        self.placeholder_button = QPushButton("原有功能占位区域")
        placeholder_layout.addWidget(self.placeholder_button)
        right_layout.addWidget(placeholder_frame)

        right_layout.addStretch()
        main_layout.addWidget(right_frame)

        # 保留菜单栏，后续可扩展更多功能
        self.createMenuBar()

    def createMenuBar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("文件")
        open_action = QAction("打开文件夹", self)
        open_action.triggered.connect(self.openFolderDialog)
        file_menu.addAction(open_action)

        help_menu = menu_bar.addMenu("帮助")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.showAbout)
        help_menu.addAction(about_action)

    def populateWorkingArea(self):
        """遍历工作区文件夹，列出所有支持格式的图片"""
        if not os.path.exists(self.working_folder):
            QMessageBox.critical(self, "错误", f"工作区文件夹不存在:\n{self.working_folder}")
            return
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        self.list_widget.clear()
        for filename in os.listdir(self.working_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                item = QListWidgetItem(filename)
                self.list_widget.addItem(item)

    def onWorkingItemClicked(self, item):
        """点击左侧工作区中的图片后，在预览区域显示该图片"""
        filename = item.text()
        image_path = os.path.join(self.working_folder, filename)
        self.selected_image_path = image_path
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled_pixmap)
        else:
            self.preview_label.setText("无法加载图片")

    def onConfirmSelection(self):
        """点击确认按钮后，处理选中图像"""
        if self.selected_image_path:
            QMessageBox.information(self, "图像选择", f"已选择图像:\n{self.selected_image_path}")
        else:
            QMessageBox.warning(self, "警告", "请先选择一张图片！")

    def openFolderDialog(self):
        """原有的文件夹选择功能"""
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹", "")
        if folder:
            QMessageBox.information(self, "文件夹选择", f"选择的文件夹:\n{folder}")

    def showAbout(self):
        QMessageBox.about(self, "关于", "平行锥束CT模拟软件\n版本 0.1\n开发者: Your Name")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
