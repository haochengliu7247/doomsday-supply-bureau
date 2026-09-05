# ComfyUI Workflows

这里保存固定版本的 API-format workflow。当前 4B graph 已完成源码对照和静态
contract test；只有完成下列全部步骤后，才标记为本机实测通过。

`flux2_klein_4b_api.json` 的实机验收清单：

1. [x] 官方 ComfyUI 版本已固定。
2. [x] 三个模型资产的大小和 SHA256 已验证。
3. [x] API graph 已由本机 ComfyUI `/prompt` 接受。
4. [x] 单参考图真实推理成功，输出可解码且不同于输入。
5. [x] 输入图、Prompt、Seed 和输出节点的语义映射已有 contract test。
6. [x] Hybrid img2img `denoise=0.9` 与接近报废的风化 Prompt 已有 contract test。
7. [x] Qwen BEFORE / AFTER 结构、灾损强度复核，拒绝重试和显存释放顺序已有测试。

`flux2_klein_4b_text_to_image_api.json` 用于没有照片、只有物品描述的情况：

1. [x] 使用相同的 4B FP8、Qwen 3 4B text encoder 和 FLUX.2 VAE。
2. [x] 使用官方 distilled 参数：完整四步调度、Euler、CFG 1.0。
3. [x] 使用 1024×1024 `EmptyFlux2LatentImage`，一次只生成一张图。
4. [x] API graph 已由本机 ComfyUI `/prompt` 接受并完成真实文字生图。
5. [x] Prompt、Seed 和输出节点的语义映射已有 provider 与 contract test。
6. [x] 不包含 `DSB_INPUT`，不会上传或伪造原始证物。

不要直接把上游 UI/subgraph workflow 当作 /prompt API graph。

固定运行时：

- ComfyUI v0.34.0 / frontend v1.49.6。
- 官方模板 `image_flux2_klein_image_edit_4b_distilled.json`，
  workflow-templates v0.11.48。
- `flux-2-klein-4b-fp8.safetensors`：4,070,624,520 bytes，
  SHA256 `97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6`。
- `qwen_3_4b.safetensors`：8,044,982,048 bytes，
  SHA256 `6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a`。
- `flux2-vae.safetensors`：336,213,556 bytes，
  SHA256 `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5`。

API graph 的唯一标记：`DSB_INPUT`、`DSB_PROMPT`、`DSB_SEED`、
`DSB_OUTPUT`。Provider 通过这些标题注入图片、提示词、随机种子和输出前缀。
文字生图 graph 只使用后三个标记，不包含图片输入标记。

身份保持改动：`VAEEncode(9)` 同时连接 ReferenceLatent 和
`SamplerCustomAdvanced.latent_image`；`Flux2Scheduler(16)` 经
`SplitSigmasDenoise(20, denoise=0.9)` 的 `low_sigmas` 进入采样器。不要改回
`EmptyFlux2LatentImage`，否则小接口和按键更容易被重新设计。输入缩放的
`resolution_steps=16`，使宽高在 VAE 下采样前对齐，避免额外的隐式裁切。
