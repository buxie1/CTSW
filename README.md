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


## 致谢
- 部分代码参考自 [Cooridnate-based-Internal-Learning](https://github.com/wustl-cig/Cooridnate-based-Internal-Learning) (CC BY-NC-SA 4.0)
- 测试数据源自 [AAPM Low Dose CT Grand Challenge](https://www.aapm.org/GrandChallenge/LowDoseCT/)
