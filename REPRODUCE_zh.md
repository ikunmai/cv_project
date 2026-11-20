# 晶体计数：团队安装与复现指南

本文针对 [ikunmai/cv_project](https://github.com/ikunmai/cv_project)，记录团队已运行的 CountVid 晶体伪示例基线和质量筛选实验。原作者说明见 [README.md](README.md)，原项目为 [CountVid](https://github.com/niki-amini-naieni/CountVid)。保留原作者 LICENSE 与署名。

## 1. 已验证的范围

- 系统：Windows + WSL2 Ubuntu 26.04.1，NVIDIA GeForce RTX 5070 Ti Laptop GPU。
- 环境：Python 3.10.22、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、CUDA Toolkit 12.8、GCC/G++ 13.4。
- NumPy 1.26.4、Transformers 4.39.1、Matplotlib 3.9.2、setuptools 75.8.0。
- SAM 2 源码提交：`2b90b9f5ceec907a1c18123530e92e794ad901a4`，使用 SAM 2.1 Large 权重。
- Detectron2 源码提交：`1e3e13bbf607b54f62205c4c33922521822fb298`，附下文的 iopath 兼容修改。

这些是实际运行机器的记录。下面的全新安装流程按此环境整理，尚未在另一台空白机器上从头验证；不同显卡、驱动和包解析结果可能需要调整。单视频已有成功运行记录，不代表完成整个晶体数据集或严格复现论文表格。

## 2. 系统准备

Windows 用户先安装支持 NVIDIA GPU 的 WSL2/Ubuntu，并在 Windows 安装适合显卡的 NVIDIA 驱动。不要在 WSL 中另装 Linux 显卡驱动。Linux 用户直接在 Linux 终端操作。

先运行：

```bash
nvidia-smi
```

应能识别 NVIDIA 显卡。本文后续命令都在 Ubuntu/Linux 的 Bash 终端执行，不是在 PowerShell 执行。准备好 Miniforge 或 Conda，确保 `conda` 可用。

```bash
sudo apt update
sudo apt install -y git build-essential libgl1 libglib2.0-0
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/ikunmai/cv_project.git
cd cv_project
```

如果已克隆，进入自己的项目目录即可，不必重复克隆。以下将仓库目录称为“项目根目录”；原开发机器目录名是 CountVid，新克隆默认是 cv_project。

## 3. 创建 Python 与 CUDA 编译环境

以下适用于支持 CUDA 12.8 的 NVIDIA 驱动与显卡。不同硬件应选择匹配的 PyTorch/CUDA 组合，不能只修改其中一个版本；本文的 constraints 文件也要相应调整。

```bash
conda create -n countvid python=3.10 -y
conda activate countvid
conda install -c conda-forge 'gcc_linux-64=13' 'gxx_linux-64=13' 'cuda-toolkit=12.8' libstdcxx-ng -y

python -m pip install 'setuptools==75.8.0' wheel ninja
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-local.txt -c constraints-local.txt

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export MAX_JOBS=2
nvcc --version
"$CC" --version
```

不要随后执行 `pip install -r requirements.txt`：原作者文件固定了旧版 torch/torchvision，会改变这套环境。需要重新编译扩展时，重新激活环境并设置上述编译变量。`nvidia-smi` 显示的 CUDA 版本是驱动支持信息，`nvcc --version` 才是这里使用的编译器版本。

GPU 检查：

```bash
python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'PyTorch 没有识别到 CUDA，请先检查驱动和安装版本'
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(128, 128, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('GPU calculation OK:', y.shape)
PY
```

## 4. 安装固定版本的 SAM 2

在项目根目录执行；这里的绝对路径由当前目录取得，不需要手动写用户名。

```bash
export COUNTVID_PROJECT_DIR="$PWD"
cd ..
git clone https://github.com/facebookresearch/sam2.git
cd sam2
git checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
python -m pip install --no-build-isolation -e . -c "$COUNTVID_PROJECT_DIR/constraints-local.txt"
cd "$COUNTVID_PROJECT_DIR"
```

如果已有 sam2 源码目录，先确认是否有自己的修改，再决定是否切换提交；不要重复执行 clone。

## 5. 编译 GroundingDINO 扩展

```bash
cd models/GroundingDINO/ops
python setup.py build --force install
python test.py
cd ../../..
```

测试应出现六项 `True`。仓库继承了作者的一些构建产物，但它们不能代替本机编译。项目 CUDA 源码已包含适配新版 PyTorch 的 `value.scalar_type()` 修改。

## 6. 安装 Detectron2，处理 iopath 冲突

SAM 2 要求 iopath >= 0.1.10，而这里固定的 Detectron2 原版要求 iopath < 0.1.10。原开发环境使用 iopath 0.1.10，并修改了 Detectron2 的依赖声明；以下复现这个兼容处理，不修改模型算法。

在项目根目录执行：

```bash
export COUNTVID_PROJECT_DIR="$PWD"
cd ..
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
git checkout 1e3e13bbf607b54f62205c4c33922521822fb298

python - <<'PY'
from pathlib import Path
p = Path('setup.py')
s = p.read_text()
old = '"iopath>=0.1.7,<0.1.10"'
new = '"iopath==0.1.10"'
if old in s:
    assert s.count(old) == 1
    p.write_text(s.replace(old, new))
else:
    assert new in s, '源码与预期不同，请先核对提交号'
PY

python -m pip install --no-build-isolation . -c "$COUNTVID_PROJECT_DIR/constraints-local.txt"
cd "$COUNTVID_PROJECT_DIR"
python -c "import detectron2; from detectron2 import _C; print('Detectron2 extension OK')"
python -m pip check
```

应看到扩展加载成功，且 `pip check` 无依赖冲突。此声明修改是本地已使用的兼容方案，不是上游对所有组合的兼容承诺。

再检查 SAM 2.1 配置：

```bash
python - <<'PY'
import sam2
from hydra import compose
print('SAM2:', sam2.__file__)
compose(config_name='configs/sam2.1/sam2.1_hiera_l.yaml')
print('SAM 2.1 config OK')
PY
```

sam2 导入时已经初始化 Hydra；这里不要额外调用 `initialize_config_module`。

## 7. 下载权重

代码仓库不包含大体积权重。在项目根目录执行：

```bash
mkdir -p checkpoints
python -m pip install gdown
gdown 1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD -O checkpoints/countgd_box.pth
curl -L --fail https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -o checkpoints/sam2.1_hiera_large.pt
python download_bert.py
ls -lh checkpoints
```

CountGD-Box 也可通过 [官方 Google Drive 链接](https://drive.google.com/file/d/1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD/view?usp=sharing) 在浏览器下载。若 gdown 提示限流或权限错误，用浏览器下载后放到相同路径，不要把错误网页当作权重文件。

目录应包含：

```text
checkpoints/
├── countgd_box.pth
├── sam2.1_hiera_large.pt
└── bert-base-uncased/
```

## 8. 下载晶体数据和标注

打开作者提供的 [VideoCount 下载链接](https://drive.google.com/file/d/1elk1-83dstQ0hEPLctrLKUNz4qUkrdDs/view?usp=sharing)，下载并解压。保持原始视频子目录与文件名，不合并帧、不自行重采样。

```text
cv_project/
└── VideoCount/
    └── Crystals/
        ├── anno/
        │   └── crystals-count-gt.json
        ├── frames/
        │   ├── ma2035_023/
        │   ├── ma2035_024/
        │   └── ...
        └── exemplars/
```

先只运行 ma2035_023 时，必须有官方标注 JSON 和 frames/ma2035_023 的完整 JPEG 帧；当前文本加伪示例模式不依赖人工 exemplars，但建议保留数据包结构。

```bash
ls VideoCount/Crystals/anno/crystals-count-gt.json
ls VideoCount/Crystals/frames
```

ma2035_023 的已验证运行日志使用 37 张 JPEG，提示词 `white crystal`，真实总数为 111。目录中其他文件不算输入帧。数据许可与使用范围以原数据发布方要求为准。

## 9. 先做预检查，再运行单视频实验

进入项目根目录，激活 `countvid` 环境。预检查不会启动模型，但需要数据和两个权重文件已在预期位置：

```bash
python run_crystals_single_quality.py --variant baseline --dry_run
python run_crystals_single_quality.py --variant quality --dry_run
```

原版算法基线：

```bash
python run_crystals_single_quality.py --variant baseline --video ma2035_023
```

只加入伪示例质量筛选的实验：

```bash
python run_crystals_single_quality.py --variant quality --video ma2035_023
```

不要同时运行两个实验。单视频也可能需要几十分钟或更久，时间过滤和跟踪是主要开销。当前没有阶段缓存或断点续跑；关机后重启命令会从头开始。

运行器固定以下对比设置：SAM 2.1 Large、CountGD-Box 相同权重、图像 batch=1、对象 batch=30、downsample_factor=1、temporal_filter 开启、w=3、convert_to_rgb 开启。它不导出视频，因此不会逐张弹窗；不要把抽帧或关闭过滤后的实验与这些结果直接混比。

两个版本的区别：

| 项目 | baseline | quality 默认值 |
|---|---|---|
| 计数入口 | count_in_videos_pseudo_exemplars.py | count_in_videos_pseudo_exemplars_quality.py |
| 伪示例置信度门槛 | > 0.23 | > 0.30 |
| 伪示例小框过滤 | 无新增规则 | 模型变换后输入图上，宽和高至少 2 像素 |
| 伪示例重复框过滤 | 无新增规则 | NMS，IoU 阈值 0.80 |
| 伪示例上限 | 10 | 10 |
| 最终检测置信度门槛 | 0.23 | 0.23 |
| 新增长期模板记忆 | 无 | 无 |

质量筛选只作用于伪示例选择，不直接对交给 SAM 的最终检测框执行新增 NMS。没有合格伪示例时返回空集合，不用被拒绝的框填充。

计划中的置信度消融实验（尚无验证结果）：

```bash
python run_crystals_single_quality.py --variant quality --video ma2035_023 --quality_min_confidence 0.23
```

这仍然保留小框过滤和 NMS，不能称为原版。

## 10. 输出与指标

每次运行自动创建独立目录，不覆盖已有结果：

```text
results/crystals_<variant>_ma2035_023_<timestamp>/
├── ground-truth.json
├── predicted.json
├── run.log
├── settings.json
└── metrics.txt
```

完成标准：子进程成功结束、预测文件包含该视频与提示词、生成 metrics.txt，且 settings.json 的 status 为 complete。只有 `{}` 的 predicted.json 不能用于评估。突然断电可能来不及写入 status，缺少 status 也不代表成功。

运行器会自动调用仓库的 evaluate_counting_accuracy.py，针对所选视频计算 MAE 和 RMSE。单视频只有一个计数样本时，两者都等于绝对误差。评估目标是全视频不同对象的累计数量，不是对每帧数量求平均。

已观察结果（记录日期：2026-11-20）：

| 版本 | 视频 | 真实数量 | 预测数量 | MAE | RMSE | 相对误差 |
|---|---|---:|---:|---:|---:|---:|
| 原算法基线 | ma2035_023 | 111 | 100 | 11 | 11 | 9.91% |
| 质量筛选默认设置 | ma2035_023 | 111 | 98 | 13 | 13 | 11.71% |

基线来自此前官方批测试已成功保存的 ma2035_023 预测；quality 来自团队单视频运行器。当前质量筛选没有改善这个样本，不应写成“已提升准确率”。不同环境和底层算子可能带来数值差异；运行器没有新增固定随机种子或确定性保证。保留每次的参数、源码哈希和原始输出；更多视频上的表现需要另测。

README 中说明 CountGD++ 论文的晶体伪示例实验使用 SAM 2，本仓库运行的是 SAM 2.1，因此不能直接宣称与该论文设置完全相同。

## 11. 常见问题

- `CUDA is available: False`：先停止实验，检查驱动、PyTorch CUDA 版本和当前 Conda 环境。
- `nvcc: command not found`：PyTorch 的 CUDA runtime 不能代替 CUDA 编译工具包；核对第 3 节。
- `MultiScaleDeformableAttention` 导入失败：在当前环境重新编译并运行六项测试，不复用其他机器的 `.so`。
- Detectron2 `_C` 无法导入：先离开 detectron2 源码目录再检查；若仍失败，核对编译日志和 PyTorch/CUDA 是否匹配。
- iopath 依赖冲突：按第 6 节对固定 Detectron2 提交打兼容修改，不用忽略所有依赖的方式跳过检查。
- `GlobalHydra is already initialized`：按第 6 节直接 compose，不重复初始化。
- `torch.load` 报 `argparse.Namespace`：项目两个原计数入口及 quality 副本已使用 safe_globals 兼容加载；检查是否拿到了团队修改后的版本，不要随意关闭安全加载。
- 无显示服务器的远程主机：当前计数脚本设置 TkAgg。单视频评估不调用预览，但 GUI 后端仍可能因环境而失败；可以在自己的副本中把 `matplotlib.use("TkAgg")` 改为 `matplotlib.use("Agg")` 后重试，并记录这一显示设置变化。
- `tostring_rgb` 弃用提醒或 MP4V fallback：这些是视频导出相关提示；本文单视频指标运行器不生成视频。
- 自动关机：不是正常的实验结束。现有日志不能判定原因，应另查系统稳定性、供电与温度；重复启动不会接着上次帧数继续跑。

## 12. 哪些文件上传 GitHub

上传源码、说明、依赖文件。模型、VideoCount 数据、临时目录、原始运行日志和结果目录由 .gitignore 排除。需要分享结果时可以把不含敏感信息的汇总表写进说明；不要上传访问 Token 或本机凭据。

更新文档时只暂存明确需要的文件，避免把上游已经跟踪的本机构建产物一起提交；.gitignore 不会自动排除已经被 Git 跟踪的文件。
