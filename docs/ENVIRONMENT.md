# SteadyMem 工作区与远端算力

## 工作方式

项目工作区为 `/mnt/bn/ecom-cyq-redirect/liuzhonglie/SteadyMem`。代码、配置、文档和实验记录在共享工作区中编写；远端服务器提供实验算力。2026 年 10 月 13 日通过 SSH 确认远端可以进入该目录，并读取与本地 SHA-256 一致的原稿快照。

```bash
cd /mnt/bn/ecom-cyq-redirect/liuzhonglie/SteadyMem
ssh -i ~/.ssh/id_merlin_managed_rsa -p 11547 \
  tiger@fdbd:dccd:cdc2:12d1:0:1fc::
```

进入远端后同样使用：

```bash
cd /mnt/bn/ecom-cyq-redirect/liuzhonglie/SteadyMem
```

用户提供的远端挂载列表包括 `chobits-wx`、`ecom-cyq-redirect`、`junting-dev`、`luoruipu-disk-2` 和 `magellan-product-audit`。项目使用 `ecom-cyq-redirect` 下的上述目录；密钥内容不进入工作区。

## 已核查硬件与共享路径

本次只读检查时间为 **2026 年 10 月 13 日 17:03:23，Asia/Shanghai**（UTC 09:03:23）。原始输出见 [环境检查记录](sources/environment-20261013.json)。

| 项目 | 检查结果 |
| --- | --- |
| 远端 hostname | `n124-161-064` |
| SSH 用户与端口 | `tiger`，`11547` |
| GPU | 8 张 `NVIDIA H100 80GB HBM3` |
| 单卡报告总显存 | `81559 MiB` |
| 检查时显存占用 | 约 `7345–9259 MiB` / 卡 |
| 检查时 GPU 利用率 | `91%–99%` |
| 共享文件验证 | `docs/sources/feishu-kda-rev12.md` 的 SHA-256 与本地一致 |

GPU 利用率与占用是瞬时状态，不表示实验吞吐或剩余可用计算资源。CUDA、驱动、PyTorch、Triton、通信与数据吞吐尚未完成项目级基准核查。

## 占卡进程

此前检查并在上述时刻再次确认 PID `4285` 仍存在，用户为 `tiger`，命令为：

```text
python3 /mnt/bn/junting-dev/liuzhonglie/gpu_occ2.py --t 1 --gpus 0,1,2,3,4,5,6,7 --per-gpu-batch 320
```

此前读取脚本确认其持续生成随机图片，使用 ResNet50 和 DataParallel 运行推理，用于占用 GPU。启动链路包含 `/mnt/bn/junting-dev/Scripts/train_valley_init/train.sh` 和 `nohup`；日志输出位置为 `/opt/log/tiger/stdout`、`/opt/log/tiger/stderr`。本次只复查进程与 GPU 状态，没有修改或停止该服务。

## 实验执行记录要求

后续实验开始前重新检查 GPU 进程和可用资源，并把运行命令、代码版本、环境版本、配置、种子、开始结束时间及输出位置保存在本项目中。占卡服务的处理应依据届时用户指令与已有授权。

8 张 H100 是资源条件，不是训练速度承诺。先测目标模型的吞吐、峰值内存和多卡效率，再估计 token 预算与运行时间；原稿的粗略 GPU 小时估计见 [实验设计](EXPERIMENTS.md)，不得视为实测。
