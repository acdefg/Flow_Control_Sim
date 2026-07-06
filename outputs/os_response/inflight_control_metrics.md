# Inflight Control Metrics

Protected variable: `sender_noc_os_gap = Sender OS - NOC OS`; target = `16` requests.

| 指标 | 控制理论含义 | 为什么重要 | NOC Watermark | Sender Outstanding | Smith PI |
| --- | --- | --- | ---: | ---: | ---: |
| Stability（稳定性） | 最终是否收敛而不是发散 | 最基本要求，没有稳定性其他都没有意义 | 稳定 | 稳定 | 稳定 |
| Steady-state Error（稳态误差） | 最终 inflight 与目标值的偏差 | 反映长期是否贴近保护目标 | -1.45 req | -1.45 req | -2.13 req |
| Overshoot（超调） | 最大 inflight 超过目标的量 | 超过目标意味着 inflight/buffer 风险 | 0.00 req | 0.00 req | 0.00 req |
| Settling Time（稳定时间） | receiver 恢复后进入最终稳定范围所需时间 | 决定扰动后的恢复速度 | 82 cyc | 220 cyc | 未在判据内稳定 |
| Rise/Action Time（响应时间） | receiver 降速后发送带宽首次下降时间 | 决定系统是否足够敏捷 | 128 cyc | 212 cyc | 10 cyc |
| Damping（阻尼） | 降速区间 inflight 振荡峰峰值 | 阻尼不足会来回开关震荡 | 8.27 req | 5.00 req | 2.39 req |
| Robustness（鲁棒性） | 参数变化后是否仍满足 inflight 目标并稳定 | Service Time 波动时是否还能工作 | 当前阶跃通过，需扫参确认 | 当前阶跃通过，需扫参确认 | 当前阶跃通过，需扫参确认 |
| Delay Margin（延迟裕度） | 能容忍多大的反馈/采样延迟 | Pipeline 延迟大时决定是否超调 | 单次实验无法给出，需扫 feedback delay | 单次实验无法给出，需扫 check interval | 单次实验无法给出，需扫 prediction/feedback delay |
