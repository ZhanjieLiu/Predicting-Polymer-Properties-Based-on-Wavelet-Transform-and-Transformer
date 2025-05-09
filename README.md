The code and dataset used for the article on "Predicting Polymer Properties Based on Wavelet Transform and Transformer".
1. "SMILES.csv"为本研究所使用的聚酰亚胺数据集，来源于Volgin等人于2022年在其研究中公开发表的开源数据集：https://doi.org/10.1021/acsomega.2c04649。
2. "Training_main.py"为本研究对应的主要代码，本章研究通过小波变换对聚酰亚胺（PI）摩根指纹进行多层次分解，提取低频与高频特征，增强分子表征的精度与抗噪能力。随后，结合Transformer模型的自注意力机制，进行多尺度特征的融合与长程依赖建模。此外，采用贝叶斯优化算法自适应调整小波分解层级及Transformer超参数，并结合Adam优化器、L2正则化与早停策略来防止过拟合。最后，我们依据回归模型评价指标，全面评估了预测模型的性能。
3. "bayesian_optimization_result.csv"为采用本文中的方法得到的贝叶斯优化过程中所有的超参数组合及其对应的验证集的评估结果。
4. 
