from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from src.config import Settings
from src.errors import AppError, ErrorCode, PipelineStage
from src.schemas import (
    AppraisalPayload,
    AppraisalResult,
    ImageIdentityVerdict,
    ScanRequest,
)

LOGGER = logging.getLogger("VLM.OLLAMA")

APPRAISAL_PROMPT_VERSION = "ollama-appraisal-2026-09-05.1"
IDENTITY_PROMPT_VERSION = "ollama-identity-2026-09-05.2"

SYSTEM_PROMPT = "\n".join(
    (
        "你是“末日物资鉴定局”的首席物证分析员。",
        "你的任务不是重新设计物品，而是先识别现实照片里的同一件具体物品，"
        "再评估它在指定末日环境中的用途与价值。",
        "",
        "硬性要求：",
        "1. 只输出符合给定 JSON Schema 的对象，不要 Markdown，不要解释。",
        "2. 观察和推断必须分开；不确定时降低 confidence，禁止捏造看不见的"
        "品牌、接口、内部结构或容量。",
        "3. original_features 与 preserve_features 必须具体记录轮廓、比例、"
        "颜色、材质、相机角度和构图；每个接口、按键和文字还必须写明数量、"
        "所在物体表面与相对位置；若最大可见表面没有控件，必须明确写出该表面"
        "无按钮、屏幕、面板、标志和文字。",
        "4. apocalypse_changes 只能是指定场景和时间下物理合理的老化、污染、"
        "磨损或修补。",
        "5. avoid_changes 必须明确禁止改变物品类别、几何结构、接口数量、"
        "视角、构图、背景以及凭空增加武器或科幻部件。",
        "6. apocalypse_name 必须把照片中的具体原物翻译成新的末日身份；不得等于"
        " original_item、场景标签或场景代码，也不能只复述灾难名称。",
        "7. image_edit_prompt 必须是可直接用于 FLUX 的 40 至 70 个英文单词："
        "只正向描述期望得到的同一件实物、可见结构锚点及其数量和位置、"
        "指定场景与时间造成的明显表面老化；不要堆叠负向关键词，也不要列举"
        "原图中不存在的部件。其余用户可见内容使用简体中文。",
        "8. 估值单位是升净水，风险与收益保持保守，所有必填列表都至少提供一项。",
    )
)

TEXT_SYSTEM_PROMPT = "\n".join(
    (
        "你是“末日物资鉴定局”的首席物证分析员。",
        "当前输入只有用户的物品文字描述，没有图像。请据此识别现实物品，并评估它在指定末日环境中的用途与价值。",
        "",
        "硬性要求：",
        "1. 只输出符合给定 JSON Schema 的对象，不要 Markdown，不要解释。",
        "2. 只能把用户明确写出的内容当作证据；不得声称看到了外观，"
        "不得捏造品牌、接口、内部结构或容量。",
        "3. observed_evidence 必须标明信息来自用户描述；推断出的材质、"
        "特征和用途必须保持保守并降低 confidence。",
        "4. original_features 与 preserve_features 只记录用户描述中的属性，"
        "以及该物品类别不可缺少的基本形态；不要添加未描述的控件或部件。",
        "5. apocalypse_changes 只能是指定场景和时间下物理合理的老化、污染、磨损或修补。",
        "6. apocalypse_name 必须把用户描述的现实物品翻译成新的末日身份；"
        "不得等于 original_item、场景标签或场景代码。",
        "7. image_edit_prompt 必须是可直接用于 FLUX 的 40 至 70 个英文单词："
        "描述画面中恰好一个完整物品、其已知结构和合理表面老化；不要添加武器"
        "或科幻部件。其余用户可见内容使用简体中文。",
        "8. 估值单位是升净水，风险与收益保持保守，所有必填列表都至少提供一项。",
    )
)

IDENTITY_SYSTEM_PROMPT = "\n".join(
    (
        "You are a conservative forensic image-comparison engine.",
        "The user message always contains exactly two images: image 1 is BEFORE and "
        "image 2 is the proposed AFTER edit.",
        "Compare source pixels region by region. Source pixels outrank the written feature "
        "contract whenever they disagree.",
        "Ignore dust, dirt, scratches, stains, corrosion, paint wear, discoloration, and "
        "plausible surface repair; those are allowed and must not cause rejection.",
        "Treat any added, missing, moved, or duplicated bounded opening, port, hole, slot, "
        "button, switch, display, lens, logo, label, construction seam, or attached part as a "
        "structural difference.",
        "Inspect every visible face separately, count small functional features, and record "
        "their relative locations. A new dark recess with a defined border is an opening, not "
        "surface wear.",
        "Use only the provided JSON Schema. Do not output Markdown or commentary. When visual "
        "evidence is ambiguous, set the relevant boolean false and lower confidence.",
    )
)


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._client = client or httpx.Client(
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_timeout_seconds,
        )
        self._owns_client = client is None
        self._monotonic = monotonic
        self._sleep = sleep

    def analyze(
        self,
        image_path: Path | None,
        request: ScanRequest,
    ) -> AppraisalResult:
        schema = AppraisalPayload.model_json_schema()
        format_schema = self._grammar_safe_schema(schema)
        has_image = image_path is not None
        user_prompt = self._build_user_prompt(request, schema, has_image=has_image)
        image_payload = self._encode_image(image_path)
        system_prompt = SYSTEM_PROMPT if has_image else TEXT_SYSTEM_PROMPT
        last_validation_error = ""

        for attempt in range(2):
            prompt = user_prompt
            if attempt:
                prompt += (
                    "\n上一次输出未通过 Schema 校验。请重新生成完整对象；"
                    "不得省略字段，不得输出额外字段。尤其要把 image_edit_prompt "
                    "改为 40 至 70 个英文单词的简短正向编辑描述。"
                    f"上一次的具体问题：{last_validation_error}"
                )
            body: dict[str, Any] = {
                "model": self.settings.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": prompt,
                        **({"images": [image_payload]} if image_payload else {}),
                    },
                ],
                "stream": False,
                "format": format_schema,
                "think": False,
                "keep_alive": self.settings.ollama_keep_alive,
                "options": {
                    "temperature": 0,
                    "num_ctx": self.settings.ollama_num_ctx,
                },
            }
            payload = self._post_json("/api/chat", body)
            content = payload.get("message", {}).get("content")
            if not isinstance(content, str) or not content.strip():
                last_validation_error = "missing message.content"
                continue
            try:
                appraisal = AppraisalPayload.model_validate_json(content)
                semantic_issues = self._appraisal_semantic_issues(appraisal, request)
                if semantic_issues:
                    last_validation_error = "; ".join(semantic_issues)
                    LOGGER.warning(
                        "[VLM] semantic validation failed attempt=%s issues=%s",
                        attempt + 1,
                        last_validation_error,
                    )
                    continue
                return AppraisalResult(**appraisal.model_dump())
            except ValidationError as exc:
                last_validation_error = self._validation_summary(exc)
                LOGGER.warning(
                    "[VLM] schema validation failed attempt=%s errors=%s",
                    attempt + 1,
                    len(exc.errors()),
                )

        raise AppError(
            code=ErrorCode.INVALID_MODEL_OUTPUT,
            stage=PipelineStage.VLM,
            user_message="视觉模型连续两次未返回合格鉴定卡，请重新扫描。",
            retriable=True,
            detail=last_validation_error,
        )

    def verify_identity(
        self,
        source_path: Path,
        candidate_path: Path,
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> ImageIdentityVerdict:
        schema = ImageIdentityVerdict.model_json_schema()
        format_schema = self._grammar_safe_schema(schema)
        source_payload = self._encode_image(source_path)
        candidate_payload = self._encode_image(candidate_path)
        assert source_payload is not None and candidate_payload is not None
        prompt = self._build_identity_prompt(appraisal, request, schema)
        last_validation_error = ""

        for attempt in range(2):
            attempt_prompt = prompt
            if attempt:
                attempt_prompt += (
                    "\nThe previous response failed schema validation. Return one complete "
                    "JSON object with every required field and no extra fields."
                )
            body: dict[str, Any] = {
                "model": self.settings.ollama_model,
                "messages": [
                    {"role": "system", "content": IDENTITY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": attempt_prompt,
                        "images": [source_payload, candidate_payload],
                    },
                ],
                "stream": False,
                "format": format_schema,
                "think": False,
                "keep_alive": self.settings.ollama_keep_alive,
                "options": {
                    "temperature": 0,
                    "num_ctx": self.settings.ollama_num_ctx,
                },
            }
            payload = self._post_json("/api/chat", body)
            content = payload.get("message", {}).get("content")
            if not isinstance(content, str) or not content.strip():
                last_validation_error = "missing message.content"
                continue
            try:
                return ImageIdentityVerdict.model_validate_json(content)
            except ValidationError as exc:
                last_validation_error = self._validation_summary(exc)
                LOGGER.warning(
                    "[VLM] identity schema validation failed attempt=%s errors=%s",
                    attempt + 1,
                    len(exc.errors()),
                )

        raise AppError(
            code=ErrorCode.INVALID_MODEL_OUTPUT,
            stage=PipelineStage.VLM,
            user_message="同一物体核验连续两次未返回合格结果，请单独重试 AFTER 图片。",
            retriable=True,
            detail=last_validation_error,
        )

    def unload(self) -> None:
        deadline = self._monotonic() + min(15, self.settings.ollama_timeout_seconds)
        self._post_json(
            "/api/generate",
            {
                "model": self.settings.ollama_model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
            timeout=min(15, self.settings.ollama_timeout_seconds),
        )
        # keep_alive=0 schedules an asynchronous runner shutdown. Wait until the
        # process inventory confirms that this exact model is no longer resident.
        while self._monotonic() < deadline:
            remaining = max(0.1, deadline - self._monotonic())
            payload = self._get_json("/api/ps", timeout=remaining)
            models = payload.get("models")
            if not isinstance(models, list):
                raise AppError(
                    code=ErrorCode.INVALID_MODEL_OUTPUT,
                    stage=PipelineStage.VLM,
                    user_message="Ollama 未返回有效的模型驻留状态。",
                    retriable=True,
                    detail=repr(models),
                )
            resident_names = {
                name
                for model in models
                if isinstance(model, dict)
                for name in (model.get("name"), model.get("model"))
                if isinstance(name, str)
            }
            if self.settings.ollama_model not in resident_names:
                return
            self._sleep(min(0.2, max(0, deadline - self._monotonic())))
        raise AppError(
            code=ErrorCode.TIMEOUT,
            stage=PipelineStage.VLM,
            user_message="视觉模型未能及时释放显存；为避免模型冲突，本次流程已停止。",
            retriable=True,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _post_json(
        self,
        endpoint: str,
        body: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            request_options = {"timeout": timeout} if timeout is not None else {}
            response = self._client.post(endpoint, json=body, **request_options)
        except httpx.TimeoutException as exc:
            raise AppError(
                code=ErrorCode.TIMEOUT,
                stage=PipelineStage.VLM,
                user_message="Ollama 鉴定超时，请确认模型已加载后重试。",
                retriable=True,
                detail=repr(exc),
            ) from exc
        except httpx.RequestError as exc:
            raise AppError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                stage=PipelineStage.VLM,
                user_message="无法连接本机 Ollama，请确认服务正在运行。",
                retriable=True,
                detail=repr(exc),
            ) from exc

        if response.status_code == 404:
            raise AppError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                stage=PipelineStage.VLM,
                user_message=(
                    f"Ollama 中找不到 {self.settings.ollama_model}；"
                    "请先拉取精确模型标签。"
                ),
                retriable=False,
                detail=response.text[:500],
            )
        if response.is_error:
            code = (
                ErrorCode.PROVIDER_UNAVAILABLE
                if response.status_code >= 500
                else ErrorCode.INVALID_MODEL_OUTPUT
            )
            raise AppError(
                code=code,
                stage=PipelineStage.VLM,
                user_message=f"Ollama 请求失败（HTTP {response.status_code}）。",
                retriable=response.status_code >= 500,
                detail=response.text[:500],
            )
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise AppError(
                code=ErrorCode.INVALID_MODEL_OUTPUT,
                stage=PipelineStage.VLM,
                user_message="Ollama 返回了无法解析的响应。",
                retriable=True,
                detail=repr(exc),
            ) from exc
        if not isinstance(payload, dict):
            raise AppError(
                code=ErrorCode.INVALID_MODEL_OUTPUT,
                stage=PipelineStage.VLM,
                user_message="Ollama 返回格式不正确。",
                retriable=True,
            )
        return payload

    def _get_json(self, endpoint: str, timeout: float) -> dict[str, Any]:
        try:
            response = self._client.get(endpoint, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise AppError(
                code=ErrorCode.TIMEOUT,
                stage=PipelineStage.VLM,
                user_message="Ollama 状态确认超时。",
                retriable=True,
                detail=repr(exc),
            ) from exc
        except httpx.RequestError as exc:
            raise AppError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                stage=PipelineStage.VLM,
                user_message="无法连接本机 Ollama，请确认服务正在运行。",
                retriable=True,
                detail=repr(exc),
            ) from exc
        if response.is_error:
            raise AppError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                stage=PipelineStage.VLM,
                user_message=f"Ollama 状态请求失败（HTTP {response.status_code}）。",
                retriable=response.status_code >= 500,
                detail=response.text[:500],
            )
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise AppError(
                code=ErrorCode.INVALID_MODEL_OUTPUT,
                stage=PipelineStage.VLM,
                user_message="Ollama 返回了无法解析的状态响应。",
                retriable=True,
                detail=repr(exc),
            ) from exc
        if not isinstance(payload, dict):
            raise AppError(
                code=ErrorCode.INVALID_MODEL_OUTPUT,
                stage=PipelineStage.VLM,
                user_message="Ollama 模型状态格式不正确。",
                retriable=True,
            )
        return payload

    @staticmethod
    def _grammar_safe_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """Keep structural constraints while avoiding llama.cpp repetition caps."""

        def project(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: project(child)
                    for key, child in value.items()
                    if key not in {"minLength", "maxLength"}
                }
            if isinstance(value, list):
                return [project(child) for child in value]
            return value

        projected = project(schema)
        assert isinstance(projected, dict)
        return projected

    @staticmethod
    def _encode_image(image_path: Path | None) -> str | None:
        if image_path is None:
            return None
        try:
            return base64.b64encode(image_path.read_bytes()).decode("ascii")
        except OSError as exc:
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                stage=PipelineStage.VLM,
                user_message="无法读取待鉴定图片，请重新上传。",
                detail=repr(exc),
            ) from exc

    @staticmethod
    def _build_user_prompt(
        request: ScanRequest,
        schema: dict[str, Any],
        *,
        has_image: bool,
    ) -> str:
        description = request.description.strip() or "未提供文字描述"
        evidence_instruction = (
            "请先根据照片证据识别现实物品，再完成末日物资鉴定。"
            if has_image
            else (
                "当前没有图像，只能依据用户文字描述完成概念性末日物资鉴定。"
                "不要声称观察到用户未描述的细节；observed_evidence 必须注明来源是用户描述。"
            )
        )
        return (
            f"末日场景：{request.scenario.label}（{request.scenario.value}）\n"
            f"经历时间：{request.apocalypse_years:g} 年\n"
            f"用户描述：{description}\n\n"
            f"{evidence_instruction}"
            "以下是必须严格遵守的 JSON Schema：\n"
            f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
        )

    @staticmethod
    def _build_identity_prompt(
        appraisal: AppraisalResult,
        request: ScanRequest,
        schema: dict[str, Any],
    ) -> str:
        feature_contract = " | ".join(appraisal.preserve_features)
        return (
            "Compare image 1 (BEFORE) with image 2 (AFTER). First inventory functional "
            "features on each visible face in both images, then report every structural "
            "difference. Judge structure only; allowed surface aging must not fail the edit.\n"
            f"Object identity hint: {appraisal.original_item}.\n"
            f"Source feature contract: {feature_contract}.\n"
            f"Scenario: {request.scenario.value}; elapsed years: "
            f"{request.apocalypse_years:g}.\n"
            "Also judge damage visibility. Set post_apocalyptic_damage_clearly_visible to true "
            "only when extreme, near-ruin, visually dominant post-apocalyptic degradation covers "
            "nearly every visible object surface. Minor dust, a few scuffs, or merely moderate "
            "weathering is false. Preserve structure despite this heavy surface destruction. "
            "Judge the target object rather than incidental background weathering. Set "
            "only_surface_condition_changed to true when silhouette, proportions, component "
            "layout, and functional geometry remain intact with nothing added, removed, moved, "
            "or duplicated, even if the retained surfaces show deep cracking, peeling layers, "
            "corrosion, embedded dirt, or heavy abrasion. Set it to false only when the object's "
            "geometry or parts actually changed.\n"
            "Return exactly this JSON Schema:\n"
            f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
        )

    @staticmethod
    def _validation_summary(exc: ValidationError) -> str:
        return "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['type']}"
            for error in exc.errors()[:8]
        )

    @staticmethod
    def _appraisal_semantic_issues(
        appraisal: AppraisalPayload,
        request: ScanRequest,
    ) -> list[str]:
        normalize = lambda value: "".join(  # noqa: E731
            character for character in value.casefold() if character.isalnum()
        )
        translated_name = normalize(appraisal.apocalypse_name)
        disallowed_names = {
            normalize(appraisal.original_item),
            normalize(request.scenario.label),
            normalize(request.scenario.value),
            normalize(f"{request.scenario.label}{request.scenario.value}"),
            normalize(f"{request.scenario.value}{request.scenario.label}"),
        }
        if translated_name in disallowed_names:
            return [
                "apocalypse_name must be a new translated identity for the object, "
                "not the original name or scenario label"
            ]
        return []
