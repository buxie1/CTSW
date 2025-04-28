import sys
import cv2
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
import plotly.express as px
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from skimage.metrics import structural_similarity

class PlotWindow(QMainWindow):
    def __init__(self, html_path):
        super().__init__()
        self.setWindowTitle("SSIM & PSNR 分析工具")
        self.setGeometry(100, 100, 1200, 800)

        # 创建浏览器组件
        self.browser = QWebEngineView()
        self.browser.load(QUrl.fromLocalFile(html_path))

        # 设置布局
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.browser)
        self.setCentralWidget(central_widget)

def calculate_psnr(gt, recon):
    mse = np.mean((gt.astype(float) - recon.astype(float)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse != 0 else float('inf')

def generate_plots(gt_path, recon_path):
    # 读取图像
    gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
    recon = cv2.cvtColor(cv2.imread(recon_path), cv2.COLOR_BGR2RGB)

    # 计算 SSIM
    ssim_value, ssim_map = structural_similarity(
        cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(recon, cv2.COLOR_RGB2GRAY),
        full=True
    )
    diff = (1 - ssim_map) * 255
    diff = diff.astype(np.uint8)

    # 生成热力图
    fig_heatmap = px.imshow(
        diff,
        color_continuous_scale='jet',
        labels={'color': 'SSIM 差异'},
        title=f'SSIM 差异热力图 (全局 SSIM: {ssim_value:.4f})'
    )

    # 生成3D图
    psnr_value = calculate_psnr(gt, recon)
    fig_3d = go.Figure(data=[go.Surface(z=diff, colorscale='jet')])
    fig_3d.update_layout(
        title=f'3D 误差表面 (PSNR: {psnr_value:.2f} dB)',
        scene=dict(zaxis_title='误差强度')
    )

    # 创建临时HTML文件
    temp_dir = tempfile.mkdtemp()
    html_path = os.path.join(temp_dir, "plots.html")

    # 将热力图保存为HTML文件
    fig_heatmap.write_html(html_path, include_plotlyjs='cdn')

    # # 将3D图保存为HTML文件
    # fig_3d.write_html(html_path, include_plotlyjs='cdn')

    return html_path



if __name__ == "__main__":
    # 生成图表并获取HTML路径
    html_file = generate_plots("F:/CTSW/ct_gt/img_4.png", "F:/CTSW/recon.png")

    # 启动应用
    app = QApplication(sys.argv)
    window = PlotWindow(html_file)
    window.show()
    sys.exit(app.exec_())
