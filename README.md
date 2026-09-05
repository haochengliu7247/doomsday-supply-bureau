# 末日物资鉴定局

> 当文明消失，万物都需要被重新翻译。

《末日物资鉴定局》把玩家现实中拍摄的具体物品，翻译成末日世界中同一件
物品的形态、身份、用途和价值，再把物资放进一个轻量生存循环。

## Project Overview

核心流程：

1. 用户拍摄、上传或仅用文字描述现实物品。
2. VLM 分析物品、材质与用途，并输出严格的末日鉴定 JSON。
3. 有照片时，Image Provider 生成经历指定灾难与时间后的同一物品；VLM 对
   BEFORE / AFTER 做双图结构复核，未通过时换 Seed 重试。
4. 只有文字时，独立的 FLUX 文生图工作流直接生成灾后物资图；BEFORE 保持为空，
   因为没有用户提供的原始证物，也不伪造“同一实物”核验结果。
5. 拒绝的照片改造候选不进入物资卡。
6. UI 展示 BEFORE / AFTER 和末日物资鉴定卡。
7. 玩家选择使用、入库、出售或放弃；也可以在 20 件固定目录中随机出现的
   废土市场商品里交易。

当前本机已完成 Phase 1–3 的 Real Mode 部署：Ollama 负责视觉鉴定，
ComfyUI + FLUX.2 Klein 4B distilled FP8 负责同物体图像编辑。安全的
`.env.example` 仍保留 Mock Mode，便于在没有模型的机器上启动。

## Game Concept

当前简化规则：

- 初始状态：生命 100、体力 80、饱食度 80、士气 70、净水 10.0L。
- 只有鉴定和休息推进一天。使用、入库、出售、放弃、购买、切换存档与补图
  都不推进时间。每天结算净水 -0.5L、体力 -3、士气 -1；鉴定还会降低
  4 点日常饱食度，休息不会降低饱食度。
- 鉴定另需体力 -5、饱食度 -6、士气 -1。饱食度不高于 20 时，鉴定额外
  体力 -4；士气低于 20 时，扫描额外体力 -2。
- 休息恢复体力 +30，并直接进入下一天。
- 废土每天都有不固定的毒气伤害，当前页面会提前明确显示下一天将损失的
  生命值；每次生命下降会触发红色扣血反馈。
- 士气相对最近高点累计每下降 5 点，整页会强烈抖动并落下尘土。
- 非食物不能恢复饱食度。扫描生成的食物只恢复 8 点；经历超过 3 个月时，
  使用会额外损失 5 点生命。
- 废土市场目录固定 20 件，其中恰好 3 件食物、3 件医疗用品，另有生存用品、
  书本、玩具和无用杂物。休息或鉴定进入新一天时随机刷新 5 件；购买后从本轮
  列表移除并进入共用 6 格背包，手动使用时才结算效果；市场物资不可转售。
  购买和使用都不推进时间。食物和医疗用品耗水较多。
- 只有经过鉴定的玩家物资能够出售。D/C/B/A/S 级分别换取
  0.1/0.5/2.0/5.0/15.0L 净水；E/F 级只换取 3/2 点士气。
- 背包用图片卡片直接选中物品，可使用、出售或放弃，不再使用下拉框。
- 页面支持多个自定义名称的本地存档；存档管理默认折叠，显示当前角色与天数。
  可切换或直接重开当前存档；永久删除需要二次确认，并始终至少保留 1 个存档。
- 结算后饱食度不高于 5：生命 -10；当日净水不足 0.5L：生命 -10。
- 生命降至 0 后结束本次生存记录。

## Architecture

依赖方向：

    Gradio UI
        |
        v
    Application Pipeline / Services
        |          |          |
        v          v          v
    VLM Provider  Image Provider  SQLite Saves / Market
        |
        v
    Domain Schemas + Pure Game Engine

目录职责：

- src/schemas.py：严格领域数据与 AI 输出结构。
- src/game/：不依赖模型、UI 或数据库的确定性规则。
- src/providers/：Mock、Ollama、ComfyUI 等可替换适配器。
- src/pipeline.py：扫描流水线和部分成功边界。
- src/services/game_repository.py：命名存档、市场目录与状态持久化。
- src/services/scan_cache_repository.py：照片/纯文字分离的完整结果精确缓存、
  词条别名与批量生成检查点。
- src/data/market_catalog.py：20 件市场商品的固定元数据和图片映射。
- src/data/common_items_zh_v1.json：按 16 个类别配额整理的 1000 个常用物品词条。
- src/ui/：Gradio 页面与 HTML 渲染。
- assets/demo/：Mock Mode 的本地示例图。
- assets/market/：预先生成并随项目保存的市场商品图。
- workflows/：固定版本的 ComfyUI API-format workflow。
- tests/：规则、Schema、Provider contract 与 Pipeline 测试。

## Environment Requirements

- Windows 11 或现代 Linux。
- Python 3.12。
- uv。
- Real Mode：NVIDIA GPU、Ollama、ComfyUI。
- 当前本机预览端口为 7861，因为 7860 已被另一个本地程序使用。

不需要为 Mock Mode 安装 CUDA Toolkit、PyTorch、Ollama 或 ComfyUI。

## Local Development

首次准备：

    uv sync --python 3.12

启动：

    uv run python app.py

Windows 也可双击：

    start.bat

访问：

    http://127.0.0.1:7861

运行测试：

    uv run pytest
    uv run ruff check .

## Mock Mode

`.env.example` 默认配置：

    MOCK_MODE=true
    VLM_PROVIDER=mock
    IMAGE_PROVIDER=mock

Mock VLM 返回与真实 Provider 相同的 Pydantic 数据结构。Mock Image Provider
只添加确定性的本地视觉磨损，用于验证 UI、状态系统和错误边界，不代表最终
FLUX 输出质量。

## Ollama Setup

使用精确 tag：

    qwen3-vl:8b-instruct-q8_0

不得使用 qwen3-vl:8b 或 latest。

本机模型目录固定为 `C:\AI\OllamaModels`。需要手动启动服务时，使用：

    start-ollama.bat

该脚本明确绑定 `127.0.0.1:11434`，不会开放外网。本机已完成以下验收：

1. 拉取精确 tag。
2. 验证 capability 包含 vision。
3. 验证结构化 JSON 连续通过。
4. 验证 ollama ps 使用 RTX A5000。
5. Image Edit 前通过 keep_alive=0 释放 VLM。

当前工作目录已经切换为：

    MOCK_MODE=false
    VLM_PROVIDER=ollama
    IMAGE_PROVIDER=comfyui

真实 demo 鉴定已通过严格 Pydantic 校验；`ollama ps` 显示 100% GPU，随后
可由 `keep_alive=0` 完整卸载。Qwen 还会在 FLUX 释放显存后复核生成前后双图，
不会与图像模型同时常驻显卡。

## ComfyUI Setup

本地技术路线为官方 Windows Portable 和 FLUX.2 Klein 4B distilled FP8，
通过 COMFYUI_BASE_URL 调用，不在业务代码中写死地址。4B 主模型可匿名下载、
采用 Apache-2.0，并且比 9B 更适合本机 RTX A5000 24GB。

需要从官方 UI workflow 导出 API-format JSON，再固定到 workflows/；
Python 只负责上传图片、注入 Prompt、排队、等待、取回输出和错误转换。

固定模型组合：

- `models/diffusion_models/flux-2-klein-4b-fp8.safetensors`
- `models/text_encoders/qwen_3_4b.safetensors`
- `models/vae/flux2-vae.safetensors`

三个文件均可匿名下载。主 4B FP8 模型是 Apache-2.0；官方工作流引用的
`flux2-vae` 所在仓库仍带非商业许可约束。因此当前组合用于本地、非公开运行，
不要把“无需登录”理解成整个模型栈都没有许可条件。

安装器会断点续传并核对文件大小与 SHA-256，不需要 Hugging Face 账号：

    powershell -ExecutionPolicy Bypass -File .\scripts\install_comfyui.ps1

安装完成后启动本机 API：

    start-comfyui.bat

默认安装位置为 `C:\AI\ComfyUI_windows_portable`，默认只监听
`127.0.0.1:8188`。

为优先保留同一件实物，正式 workflow 在官方 ReferenceLatent 编辑流上增加了
identity-preserving img2img：输入图 VAE latent 同时作为采样起点，并通过
`SplitSigmasDenoise` 固定使用 `denoise=0.9`，运行完整四步调度。Prompt 要求
接近报废、覆盖几乎全部可见表面的灾后风化，同时禁止缺件、穿孔、变形和新增组件；
Qwen 双图门禁会同时拒绝结构走样和灾后损伤不明显的候选。

本机 API workflow 已完成模型哈希、`/prompt`、真实单图推理、输出解码和双图
核验验收。另有独立的四步 text-to-image workflow：纯文字输入会直接生成一张
1024×1024 灾后物资图，不经过上传接口，也不会把 AI 图片冒充 BEFORE 证物。
生成后的两个模型都会主动释放显存。

8B VLM 对非常小的局部结构并非绝对可靠，所以双图核验是额外门禁，不应理解为
数学意义上的身份保证。真正的主要保真措施是以原图 latent 为采样起点并只运行
完整四步调度；页面只展示通过自动核验的候选，关键演示图仍建议人工目测
一次。

- 4B FP8 模型卡：https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8
- 官方 ComfyUI 指南：https://docs.comfy.org/tutorials/flux/flux-2-klein

## Configuration

全部配置来自 .env：

- MOCK_MODE
- VLM_PROVIDER
- IMAGE_PROVIDER
- OLLAMA_BASE_URL
- OLLAMA_MODEL
- COMFYUI_BASE_URL
- COMFYUI_WORKFLOW_PATH
- COMFYUI_TEXT_WORKFLOW_PATH
- COMFYUI_MIN_FREE_VRAM_MB
- IMAGE_IDENTITY_MAX_ATTEMPTS
- IMAGE_IDENTITY_MIN_CONFIDENCE
- DATABASE_URL
- OUTPUT_DIR
- SCAN_CACHE_ENABLED
- AI_PIPELINE_LOCK_PATH
- AI_PIPELINE_LOCK_TIMEOUT_SECONDS

.env.example 可提交；.env 不进入版本控制。

## Model Switching

Provider 只向 Pipeline 返回领域 Schema，不暴露 Ollama 或 ComfyUI 的原始
响应。未来替换 ModelScope、OpenAI-compatible VLM 或远程 Image Edit 时，
不需要修改 Game Engine 和 UI 业务规则。

## Complete Result Cache

缓存保存的是一次鉴定的完整成功结果：严格结构化鉴定数据、AFTER 图片路径、
图片哈希与生成管线签名。图片仍保存为普通文件，SQLite 只存相对路径和索引，
避免把大量二进制塞入数据库。读取命中后不会启动 Ollama 或 FLUX，并会为本次
游戏生成新的背包物品 ID；失败结果、降级结果以及照片身份核验未通过的结果均
不会入库。

纯文字与照片使用不同的缓存键。纯文字键包含规范化物品描述、末日场景、时间和
完整管线签名；照片键还包含基于规范化像素计算的图片 SHA-256，所以同名但内容
不同的照片不会串用。模型 tag、Prompt 版本、工作流文件或鉴定 Schema 改变时，
签名会自动变化，旧缓存不会被误命中。所有缓存路径和文件哈希在读取时都会复核。

1000 个常用词条当前统一预生成为“城市断电 + 3 年”，任务可在任何一项后安全
中断并从 SQLite 检查点继续：

    .venv\Scripts\python.exe -m scripts.prewarm_scan_cache

查看进度（不会加载模型）：

    .venv\Scripts\python.exe -m scripts.prewarm_scan_cache --status

验证词表和数据库清单，或只试跑少量项目：

    .venv\Scripts\python.exe -m scripts.prewarm_scan_cache --validate-only
    .venv\Scripts\python.exe -m scripts.prewarm_scan_cache --limit 3

页面扫描与批量任务共用一个跨进程 GPU 锁。批处理每完成一项都会释放模型并写入
检查点，页面请求可在项目之间获得显卡，不会和后台 Ollama / FLUX 同时占用显存。

## Troubleshooting

- 端口占用：修改 .env 中的 APP_PORT。
- 页面可开但扫描失败：确认 MOCK_MODE=true，再查看 [PIPELINE] 日志。
- 图片无输出：确认格式有效且文件不超过配置限制。
- Ollama 不可用：回到 Mock Mode；不要让 UI 直接展示 traceback。
- ComfyUI OOM：保持单并发，先卸载 Ollama，再使用默认 Dynamic VRAM。
- 扫描提示显存未恢复：检查是否有另一个 Ollama / ComfyUI 服务正在占用同一张
  GPU；本项目默认至少等待 20,000 MiB 全局可用显存后才切换模型。

## Future ModelScope Deployment

ModelScope Space 无法访问开发机上的 localhost:11434 或 localhost:8188。
部署时将替换为远程 VLM/Image Provider，并把多人持久状态从本地 SQLite
迁移到外部数据库。核心翻译、规则和 UI 不依赖具体模型供应商。
