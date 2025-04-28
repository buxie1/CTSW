# CTSW
## 60秒快速部署
```bash
# 克隆仓库
git clone https://github.com/buxie1/CTSW.git && cd CTSW

# 创建环境
conda env create -f environment.yml
conda activate CTSW

# 运行演示
python src/main7.py --demo --gpu 0

三、代码可读性优化
主程序 (main7.py) 增强

添加函数级文档字符串：

python
def reconstruct_ct(input_path: str, output_dir: str, use_gpu: bool = True) -> None:
    """
    执行稀疏视角CT重建全流程
    
    Args:
        input_path:  输入投影数据路径（支持.mat/.dcm）
        output_dir: 重建图像输出目录
        use_gpu:    是否启用GPU加速（默认True）
    """
使用 argparse 模块规范命令行参数

模型代码注释示例

python
class SIRENWithFBP(nn.Module):
    """融合滤波反投影先验的SIREN网络"""
    def __init__(self, in_features=3, hidden_features=256):
        """
        Args:
            in_features: 输入维度（x,y,投影角）
            hidden_features: 隐层神经元数（默认256）
        """
        super().__init__()
        self.fbp_layer = FBPLayer()  # 物理先验模块
        self.siren = Siren(...)      # 隐式神经表示模块
