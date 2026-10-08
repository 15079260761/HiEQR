# DECGC on CronQuestions

该目录是独立的 CronQuestions 泛化实验实现。检索器、重排器、动态证据链图、推理器、损失函数和神经—符号答案聚合均继承原 DECGC。遵循官方 perfect entity linking 设定，推理输入只使用自然语言问题及官方实体/时间链接；关系、算子、方向和时间锚点由问题确定，gold relation、annotation 和 answers 不进入推理。

## 固定路径

- 代码：`cronquestions_src`
- 数据：`data/CronQuestions/wikidata_big`
- 预训练模型：`models`
- 独立输出：`cronquestions_outputs`
- Demo 配置：`cronquestions_src/configs/experiments/cronquestions_demo.yaml`
- 全量测试配置：`cronquestions_src/configs/experiments/cronquestions_full.yaml`

## 实验协议

- 按官方五类问题与 Entity/Time 答案类型进行固定种子、比例分层的 10,000 条训练采样。
- 验证集分层采样 2,000 条，只用于选择三个工具器的最佳检查点。
- 最佳检查点先在测试集固定分层的 500 条 Demo 上快速验证。
- 配置冻结后，在官方 30,000 条测试集上执行一次全量评估。
- 输出 Hits@1/Hits@10 的 Overall、Simple、Complex、Entity、Time 十项表格字段。

`quality_gate.hits1_min=0.985` 是实验目标与验收阈值，不会篡改预测或保证模型一定达到该值。真实结果始终保存。

## 断点与磁盘策略

所有耗时阶段使用 `--resume auto`，缓存清单同时校验数据签名、样本 ID、阶段配置和最佳模型。中断后重新执行同一脚本即可从有效阶段/分片继续。每个训练阶段仅保留 1 个最佳和 1 个最新检查点；切换阶段前后只清理当前配置无法续用的旧 run、临时文件和已合并分片，不删除当前最佳模型或有效缓存。

## 执行

```bash
bash cronquestions_src/scripts/install_cronquestions.sh
tmux new-session -d -s decgc_cron_demo "bash -lc 'cd \"$(pwd)\" && source .venv/bin/activate && bash cronquestions_src/scripts/run_cronquestions_demo_pipeline.sh'"
tmux attach-session -t decgc_cron_demo
```

Demo 达标并冻结配置后：

```bash
tmux new-session -d -s decgc_cron_full "bash -lc 'cd \"$(pwd)\" && source .venv/bin/activate && bash cronquestions_src/scripts/run_cronquestions_full_test.sh'"
tmux attach-session -t decgc_cron_full
```

从 tmux 脱离：先按 `Ctrl+B`，松开后按 `D`。异常中断后的续跑命令与首次命令完全相同；若 session 仍存在，直接 `tmux attach-session -t ...`。
