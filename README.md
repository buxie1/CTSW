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
