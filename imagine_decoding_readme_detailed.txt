Imagine Decoding Baseline – Detailed Slide Notes
想象解码基线 – 详细汇报版

==================================================
1. Task Overview / 任务概述
==================================================
- MEG imagined word decoding (10-class classification)
- 使用 MEG 信号解码“想象中的词语”（10分类）

Classes / 类别:
apple, bicycle, brush, cake, clown, cup, desk, foot, mountain, zebra

Goal / 目标:
- Train model on localizer data
- 在 localizer 数据上训练模型
- Predict labels for imagine data
- 对 imagine 数据进行分类预测

==================================================
2. Data Characteristics / 数据特点
==================================================
- Sampling rate: 100 Hz
- 采样率：100 Hz

- ~306 MEG channels
- 约 306 个 MEG 通道

Signal properties / 信号特点:
- Localizer: earlier, stronger, clearer
  localizer 信号更早、更强、更清晰
- Imagine: later, weaker, noisier
  imagine 信号更晚、更弱、更噪声大

Challenge / 挑战:
- Domain shift between localizer and imagine
  localizer 与 imagine 存在分布差异（迁移问题）

==================================================
3. Core Method / 核心方法
==================================================
Pipeline:
StandardScaler → (optional PCA) → LogisticRegression

Why this baseline? / 为什么用这个基线：
- Simple & stable（简单稳定）
- Fast to iterate（方便快速实验）
- Good baseline for comparison（作为后续改进对照）

==================================================
4. Feature Engineering / 特征构造
==================================================
Steps:
1. Load epochs（加载 epoch 数据）
2. Crop time window（裁剪时间窗）
3. Flatten (channels × time) → vector
   将通道 × 时间展平成向量

Result:
- Each trial → 1 feature vector
- 每个 trial 对应一个特征向量

IMPORTANT / 关键点:
- Feature dimension depends on time window length
- 特征维度取决于时间窗长度

==================================================
5. Time Window Design / 时间窗设计
==================================================
Current setting:

Train (localizer):
- 0.08 – 0.40 sec

Test (imagine):
- 0.50 – 0.82 sec

Duration:
- Both = 0.32 sec（必须相同）

Why different positions? / 为什么位置不同：
- Localizer signal appears earlier
- Imagine signal appears later

CRITICAL CONSTRAINT / 关键限制:
- Train/test window duration MUST match
- 否则特征维度不一致，会报错

==================================================
6. Training & Prediction Flow / 训练与预测流程
==================================================
For each subject:

Training:
- Load localizer epochs
- Crop to train window
- Flatten to features
- Extract labels
- Train classifier

Prediction:
- Load imagine epochs
- Crop to test window
- Flatten to features
- Predict labels

Output:
- submission.csv

==================================================
7. Model Details / 模型细节
==================================================
Logistic Regression:
- solver: lbfgs
- class_weight: balanced
- max_iter: 2000

Preprocessing:
- StandardScaler (normalize features)

Optional:
- PCA (dimensionality reduction)

==================================================
8. Current Status / 当前进展
==================================================
- Pipeline runs successfully
- 已成功运行完整流程

- submission.csv generated
- 已生成提交文件

- Ready for baseline submission
- 可以作为 baseline 提交比赛

==================================================
9. Evaluation Strategy / 评估策略
==================================================
1. Localizer CV:
- Evaluate learnability of localizer
- 判断 localizer 是否有可学习信息

2. Leaderboard:
- Evaluate real performance
- 评估真实效果

Important:
- CV ≠ Leaderboard
- CV 与排行榜不完全一致

==================================================
10. Current Limitations / 当前局限
==================================================
- Flattening loses temporal structure
  展平操作丢失时间结构

- Requires equal window length
  必须使用等长时间窗

- Domain shift not handled explicitly
  未显式处理 domain shift

==================================================
11. Next Steps / 下一步优化
==================================================
(1) PCA tuning:
- 50 / 100 / 150 / 200

(2) Window search:
- Shift windows (keep same duration)
- 调整时间窗位置（保持长度一致）

(3) Multi-window ensemble:
- Train multiple models
- Average predict_proba
- 多窗口模型融合

(4) Advanced (later):
- Temporal binning
- Separate mag / grad
- Domain adaptation

==================================================
12. Key Takeaways / 关键总结
==================================================
- Simple baseline already works
  简单模型已经可以跑通

- Main challenge = domain shift
  主要难点在分布迁移

- Incremental improvements are effective
  增量优化最有效

==================================================
