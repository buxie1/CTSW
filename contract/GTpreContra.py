import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from skimage.metrics import structural_similarity
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os


class PlotWindow(QMainWindow):
    def __init__(self, html_path):
        super().__init__()
        self.setWindowTitle("SSIM & PSNR 分析工具")
        self.setGeometry(100, 100, 1200, 800)

        self.browser = QWebEngineView()
        self.browser.load(QUrl.fromLocalFile(html_path))

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

    # 修正关键部分：直接注入完整的图表JSON
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(f"""
        <html>
            <head>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body style="margin: 0; padding: 20px;">
                <div id="heatmap" style="height: 50vh; width: 100%;"></div>
                <div id="3dplot" style="height: 50vh; width: 100%;"></div>

                <script>
                    // 热力图
                    var heatmapData = {fig_heatmap.to_json()};
                    Plotly.newPlot('heatmap', heatmapData.data, heatmapData.layout);

                    // 3D图
                    var surfaceData = {fig_3d.to_json()};
                    Plotly.newPlot('3dplot', surfaceData.data, surfaceData.layout);
                </script>
            </body>
        </html>
        """)
    return html_path


if __name__ == "__main__":
    html_file = generate_plots("F:/CTSW/ct_gt/img_4.png", "F:/CTSW/recon.png")
    app = QApplication(sys.argv)
    window = PlotWindow(html_file)
    window.show()
    sys.exit(app.exec_())