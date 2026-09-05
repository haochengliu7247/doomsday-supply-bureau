from __future__ import annotations

import json
import logging
import mimetypes
import os
import subprocess
import time
from collections.abc import Callable
from copy import deepcopy
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import httpx
from PIL import Image, ImageChops, UnidentifiedImageError

from src.config import Settings
from src.errors import AppError, ErrorCode, PipelineStage
from src.schemas import AppraisalResult, ScanRequest
from src.services.image_service import MAX_INPUT_PIXELS

LOGGER = logging.getLogger("IMAGE_EDIT.COMFYUI")

IMAGE_EDIT_PROMPT_VERSION = "comfy-edit-2026-09-05.2"
TEXT_TO_IMAGE_PROMPT_VERSION = "comfy-text-2026-09-05.1"

INPUT_MARKER = "DSB_INPUT"
PROMPT_MARKER = "DSB_PROMPT"
OUTPUT_MARKER = "DSB_OUTPUT"
SEED_MARKER = "DSB_SEED"
OOM_MARKERS = (
    "torch.outofmemoryerror",
    "cuda out of memory",
    "allocation on device",
    "xpu out of memory",
    "mps backend out of memory",
)
UNLOADED_TORCH_VRAM_LIMIT_BYTES = 512 * 1024 * 1024
SCENARIO_SURFACE_EFFECTS = {
    "CITY_BLACKOUT": (
        "thick embedded grime, dense abrasion networks, broad coating wear, pronounced "
        "edge scarring, and widespread oxidation on existing exposed metal"
    ),
    "NUCLEAR_RUINS": (
        "heavy ash crust, extensive chalked coating, dense impact scarring, and widespread "
        "non-perforating surface pitting"
    ),
    "FLOOD": (
        "thick dried silt crust, multiple dark waterlines, broad damp staining, and "
        "widespread oxidation on exposed metal"
    ),
    "EXTREME_COLD": (
        "dense frost etching, extensive non-structural cold crazing, broad coating loss, "
        "and severe brittle edge weathering"
    ),
    "EXTREME_HEAT": (
        "heavy soot crust, broad heat discoloration, extensive blistered and charred coating, "
        "and pronounced scorched edge wear"
    ),
    "NATURE_RECLAIMS": (
        "established lichen crusts, broad moss staining, dense biological patina, and "
        "widespread oxidation on exposed metal"
    ),
}


class ComfyUIProvider:
    name = "comfyui"

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        global_free_vram_mb: Callable[[], int] | None = None,
    ) -> None:
        self.settings = settings
        self._client = client or httpx.Client(
            base_url=settings.comfyui_base_url,
            timeout=settings.comfyui_timeout_seconds,
        )
        self._owns_client = client is None
        self._monotonic = monotonic
        self._sleep = sleep
        self._global_free_vram_mb = (
            global_free_vram_mb or self._query_global_free_vram_mb
        )
        self._workflow: dict[str, dict[str, Any]] | None = None
        self._text_workflow: dict[str, dict[str, Any]] | None = None

    def edit(
        self,
        image_path: Path | None,
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> Path | None:
        text_only = image_path is None
        if text_only and not request.description.strip():
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                stage=PipelineStage.INPUT,
                user_message="没有照片时，请先填写物品描述。",
            )
        if image_path is not None and not image_path.is_file():
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="找不到待编辑的原始图片，请重新上传。",
            )

        deadline = self._monotonic() + self.settings.comfyui_timeout_seconds
        template = self._load_text_workflow() if text_only else self._load_workflow()
        graph = deepcopy(template)
        prompt_id, _ = self._find_unique_marker(graph, PROMPT_MARKER)
        output_id, _ = self._find_unique_marker(graph, OUTPUT_MARKER)
        seed_marker = self._find_optional_marker(graph, SEED_MARKER)

        job_token = uuid4().hex
        if text_only:
            graph[prompt_id]["inputs"]["text"] = self._text_to_image_prompt(
                appraisal, request
            )
        else:
            input_id, _ = self._find_unique_marker(graph, INPUT_MARKER)
            remote_image = self._upload_image(image_path, job_token, deadline)
            graph[input_id]["inputs"]["image"] = remote_image
            graph[prompt_id]["inputs"]["text"] = self._prompt(appraisal, request)
        if seed_marker is not None:
            self._inject_seed(graph[seed_marker[0]], uuid4().int & ((1 << 63) - 1))
        graph[output_id]["inputs"]["filename_prefix"] = f"DSB/{job_token}"

        queue_payload = self._request_json(
            "POST",
            "/prompt",
            deadline,
            json={"prompt": graph, "client_id": job_token},
        )
        node_errors = queue_payload.get("node_errors")
        if node_errors:
            self._raise_workflow_error(
                "ComfyUI 拒绝了当前工作流，请检查模型和节点版本。",
                node_errors,
            )
        queued_prompt_id = queue_payload.get("prompt_id")
        if not isinstance(queued_prompt_id, str) or not queued_prompt_id.strip():
            self._raise_workflow_error(
                "ComfyUI 没有返回任务编号，请检查工作流。",
                queue_payload,
                retriable=True,
            )

        try:
            record = self._wait_for_history(queued_prompt_id, deadline)
            image_record = self._extract_image_record(record, output_id)
            image_bytes = self._download_image(image_record, deadline)
            return self._persist_result(image_bytes, image_path, job_token)
        except AppError as exc:
            if exc.code is ErrorCode.TIMEOUT:
                self._cancel_prompt(queued_prompt_id)
            raise

    def unload(self) -> None:
        deadline = self._monotonic() + min(15, self.settings.comfyui_timeout_seconds)
        self._request(
            "POST",
            "/free",
            deadline,
            json={"unload_models": True, "free_memory": True},
        )
        # /free only sets an asynchronous worker flag. Do not let the pipeline load
        # Ollama until the Comfy torch allocator confirms that model memory is gone.
        while self._monotonic() < deadline:
            payload = self._request_json("GET", "/system_stats", deadline)
            devices = payload.get("devices")
            if not isinstance(devices, list) or not devices:
                self._raise_workflow_error(
                    "ComfyUI 未返回可用设备状态，无法确认图像模型已释放。",
                    payload,
                    retriable=True,
                )
            cuda_devices = [
                device
                for device in devices
                if isinstance(device, dict) and device.get("type") == "cuda"
            ]
            if not cuda_devices or not all(
                isinstance(device.get("torch_vram_total"), int)
                for device in cuda_devices
            ):
                self._raise_workflow_error(
                    "ComfyUI 显存状态格式不正确，无法确认图像模型已释放。",
                    devices,
                    retriable=True,
                )
            minimum_free_vram = self.settings.comfyui_min_free_vram_mb * 1024 * 1024
            global_free_vram_mb = self._global_free_vram_mb()
            if all(
                device["torch_vram_total"] <= UNLOADED_TORCH_VRAM_LIMIT_BYTES
                for device in cuda_devices
            ) and global_free_vram_mb * 1024 * 1024 >= minimum_free_vram:
                return
            self._sleep_for_poll(deadline)
        raise AppError(
            code=ErrorCode.TIMEOUT,
            stage=PipelineStage.IMAGE_EDIT,
            user_message=(
                "GPU 可用显存未恢复到安全阈值；可能有其他本地模型正在占用，"
                "已停止以避免降速或冲突。"
            ),
            retriable=True,
        )

    @staticmethod
    def _query_global_free_vram_mb() -> int:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--id=0",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            first_line = completed.stdout.strip().splitlines()[0]
            return int(first_line.strip())
        except (
            FileNotFoundError,
            IndexError,
            ValueError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            raise AppError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                stage=PipelineStage.IMAGE_EDIT,
                user_message=(
                    "无法读取 NVIDIA 全局显存状态；为避免模型冲突，本次流程已停止。"
                ),
                retriable=True,
                detail=repr(exc),
            ) from exc

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _load_workflow(self) -> dict[str, dict[str, Any]]:
        if self._workflow is not None:
            return self._workflow
        path = self.settings.comfyui_workflow_path
        graph = self._read_workflow(path)

        input_id, input_node = self._find_unique_marker(graph, INPUT_MARKER)
        prompt_id, prompt_node = self._find_unique_marker(graph, PROMPT_MARKER)
        output_id, output_node = self._find_unique_marker(graph, OUTPUT_MARKER)
        if input_node["class_type"] != "LoadImage" or not isinstance(
            input_node["inputs"].get("image"), str
        ):
            self._raise_workflow_error("DSB_INPUT 必须标记唯一的 LoadImage 节点。", input_id)
        if not isinstance(prompt_node["inputs"].get("text"), str):
            self._raise_workflow_error("DSB_PROMPT 节点必须包含文本输入。", prompt_id)
        if output_node["class_type"] != "SaveImage":
            self._raise_workflow_error("DSB_OUTPUT 必须标记最终 SaveImage 节点。", output_id)
        self._find_optional_marker(graph, SEED_MARKER)
        self._workflow = graph
        return graph

    def _load_text_workflow(self) -> dict[str, dict[str, Any]]:
        if self._text_workflow is not None:
            return self._text_workflow
        path = self.settings.comfyui_text_workflow_path
        graph = self._read_workflow(path)

        if self._find_optional_marker(graph, INPUT_MARKER) is not None:
            self._raise_workflow_error(
                "文字生图工作流不能包含 DSB_INPUT 图片输入标记。",
                {"path": str(path)},
            )
        prompt_id, prompt_node = self._find_unique_marker(graph, PROMPT_MARKER)
        seed_id, seed_node = self._find_unique_marker(graph, SEED_MARKER)
        output_id, output_node = self._find_unique_marker(graph, OUTPUT_MARKER)
        if prompt_node["class_type"] != "CLIPTextEncode" or not isinstance(
            prompt_node["inputs"].get("text"), str
        ):
            self._raise_workflow_error(
                "文字生图的 DSB_PROMPT 必须标记 CLIPTextEncode 节点。", prompt_id
            )
        self._inject_seed(seed_node, 0)
        if output_node["class_type"] != "SaveImage":
            self._raise_workflow_error("DSB_OUTPUT 必须标记最终 SaveImage 节点。", output_id)
        self._text_workflow = graph
        return graph

    def _read_workflow(self, path: Path) -> dict[str, dict[str, Any]]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AppError(
                code=ErrorCode.IMAGE_EDIT_FAILED,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="ComfyUI API 工作流缺失或损坏，请先完成工作流安装。",
                retriable=False,
                detail=repr(exc),
            ) from exc
        if not isinstance(raw, dict) or "nodes" in raw or not raw:
            self._raise_workflow_error(
                "工作流不是 ComfyUI API Format，请重新导出。",
                {"path": str(path)},
            )
        graph: dict[str, dict[str, Any]] = {}
        for node_id, node in raw.items():
            if not isinstance(node_id, str) or not isinstance(node, dict):
                self._raise_workflow_error("ComfyUI 工作流节点格式不正确。", node_id)
            if not isinstance(node.get("class_type"), str) or not isinstance(
                node.get("inputs"), dict
            ):
                self._raise_workflow_error(
                    "ComfyUI 工作流节点缺少 class_type 或 inputs。", node_id
                )
            graph[node_id] = node
        return graph

    def _upload_image(self, image_path: Path, token: str, deadline: float) -> str:
        suffix = image_path.suffix.lower() if image_path.suffix else ".png"
        remote_name = f"dsb_{token}{suffix}"
        mime = mimetypes.guess_type(remote_name)[0] or "application/octet-stream"
        try:
            with image_path.open("rb") as handle:
                response = self._request_json(
                    "POST",
                    "/upload/image",
                    deadline,
                    files={"image": (remote_name, handle, mime)},
                    data={"type": "input", "overwrite": "false"},
                )
        except OSError as exc:
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="无法读取待编辑图片，请重新上传。",
                detail=repr(exc),
            ) from exc
        name, subfolder, remote_type = self._validated_remote_record(response)
        if remote_type != "input":
            self._raise_workflow_error("ComfyUI 上传结果类型不正确。", response)
        return f"{subfolder}/{name}" if subfolder else name

    def _wait_for_history(self, prompt_id: str, deadline: float) -> dict[str, Any]:
        while self._monotonic() < deadline:
            payload = self._request_json(
                "GET", f"/history/{prompt_id}", deadline
            )
            record = payload.get(prompt_id)
            if record is None:
                self._sleep_for_poll(deadline)
                continue
            if not isinstance(record, dict):
                self._raise_workflow_error(
                    "ComfyUI 返回了无效的任务记录。", record, retriable=True
                )
            status = record.get("status", {})
            serialized = self._detail(record)
            if self._contains_oom(serialized):
                self._raise_oom(serialized)
            status_str = status.get("status_str") if isinstance(status, dict) else None
            if status_str == "error" or "execution_error" in serialized.lower():
                raise AppError(
                    code=ErrorCode.IMAGE_EDIT_FAILED,
                    stage=PipelineStage.IMAGE_EDIT,
                    user_message="AI 图像编辑执行失败，请稍后重试。",
                    retriable=True,
                    detail=serialized,
                )
            completed = status.get("completed") if isinstance(status, dict) else None
            if completed is True or record.get("outputs"):
                return record
            self._sleep_for_poll(deadline)
        raise AppError(
            code=ErrorCode.TIMEOUT,
            stage=PipelineStage.IMAGE_EDIT,
            user_message="AI 图像编辑超时，可稍后单独重试 AFTER 图片。",
            retriable=True,
        )

    def _extract_image_record(
        self, record: dict[str, Any], output_id: str
    ) -> dict[str, Any]:
        outputs = record.get("outputs")
        output = outputs.get(output_id) if isinstance(outputs, dict) else None
        images = output.get("images") if isinstance(output, dict) else None
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
            self._raise_workflow_error(
                "ComfyUI 已完成任务，但没有返回唯一的 AFTER 图片。",
                output,
                retriable=True,
            )
        return images[0]

    def _download_image(self, record: dict[str, Any], deadline: float) -> bytes:
        name, subfolder, remote_type = self._validated_remote_record(record)
        if remote_type != "output":
            self._raise_workflow_error("ComfyUI 输出图片类型不正确。", record)
        response = self._request(
            "GET",
            "/view",
            deadline,
            params={"filename": name, "subfolder": subfolder, "type": remote_type},
        )
        content_type = response.headers.get("content-type", "").lower()
        if not content_type.startswith("image/"):
            self._raise_workflow_error(
                "ComfyUI 返回的结果不是图片。", {"content_type": content_type}
            )
        image_bytes = response.content
        limit = self.settings.comfyui_max_output_mb * 1024 * 1024
        if not image_bytes or len(image_bytes) > limit:
            self._raise_workflow_error(
                "ComfyUI 输出图片为空或超过大小限制。",
                {"bytes": len(image_bytes), "limit": limit},
            )
        return image_bytes

    def _persist_result(
        self,
        data: bytes,
        source_path: Path | None,
        token: str,
    ) -> Path:
        try:
            with Image.open(BytesIO(data)) as candidate:
                candidate.load()
                if candidate.width * candidate.height > MAX_INPUT_PIXELS:
                    raise ValueError("output pixel count exceeds safety limit")
                result = candidate.convert("RGB")
            if source_path is not None:
                with Image.open(source_path) as source:
                    source_rgb = source.convert("RGB")
                    source_rgb.load()
                if source_rgb.size == result.size and ImageChops.difference(
                    source_rgb, result
                ).getbbox() is None:
                    self._raise_workflow_error(
                        "图像模型没有产生可见变化，请单独重试 AFTER 图片。",
                        {"same_pixels": True},
                        retriable=True,
                    )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise AppError(
                code=ErrorCode.IMAGE_EDIT_FAILED,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="ComfyUI 返回了损坏的图片，请稍后重试。",
                retriable=True,
                detail=repr(exc),
            ) from exc

        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.settings.output_dir / f"dsb_{token}_after.png"
        temporary_path = self.settings.output_dir / f".{output_path.name}.tmp"
        try:
            result.save(temporary_path, format="PNG", optimize=True)
            os.replace(temporary_path, output_path)
        except OSError as exc:
            temporary_path.unlink(missing_ok=True)
            raise AppError(
                code=ErrorCode.INTERNAL,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="无法保存 AFTER 图片，请检查磁盘空间后重试。",
                retriable=True,
                detail=repr(exc),
            ) from exc
        return output_path

    def _request_json(
        self,
        method: str,
        endpoint: str,
        deadline: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = self._request(method, endpoint, deadline, **kwargs)
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise AppError(
                code=ErrorCode.IMAGE_EDIT_FAILED,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="ComfyUI 返回了无法解析的响应。",
                retriable=True,
                detail=repr(exc),
            ) from exc
        if not isinstance(payload, dict):
            self._raise_workflow_error(
                "ComfyUI 返回格式不正确。", payload, retriable=True
            )
        return payload

    def _request(
        self,
        method: str,
        endpoint: str,
        deadline: float,
        **kwargs: Any,
    ) -> httpx.Response:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise AppError(
                code=ErrorCode.TIMEOUT,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="AI 图像编辑超时，可稍后单独重试 AFTER 图片。",
                retriable=True,
            )
        try:
            response = self._client.request(
                method, endpoint, timeout=max(0.1, remaining), **kwargs
            )
        except httpx.TimeoutException as exc:
            raise AppError(
                code=ErrorCode.TIMEOUT,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="AI 图像编辑超时，可稍后单独重试 AFTER 图片。",
                retriable=True,
                detail=repr(exc),
            ) from exc
        except httpx.RequestError as exc:
            raise AppError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="无法连接本机 ComfyUI，请确认服务正在运行。",
                retriable=True,
                detail=repr(exc),
            ) from exc
        if response.is_error:
            detail = response.text[:1000]
            if self._contains_oom(detail):
                self._raise_oom(detail)
            provider_down = response.status_code in {502, 503, 504}
            raise AppError(
                code=(
                    ErrorCode.PROVIDER_UNAVAILABLE
                    if provider_down
                    else ErrorCode.IMAGE_EDIT_FAILED
                ),
                stage=PipelineStage.IMAGE_EDIT,
                user_message=f"ComfyUI 请求失败（HTTP {response.status_code}）。",
                retriable=provider_down or response.status_code >= 500,
                detail=detail,
            )
        return response

    def _sleep_for_poll(self, deadline: float) -> None:
        remaining = deadline - self._monotonic()
        if remaining > 0:
            self._sleep(min(self.settings.comfyui_poll_interval_seconds, remaining))

    def _cancel_prompt(self, prompt_id: str) -> None:
        try:
            self._client.post("/queue", json={"delete": [prompt_id]}, timeout=2)
        except httpx.HTTPError:
            LOGGER.warning("[IMAGE_EDIT] unable to remove timed-out queued prompt")
        try:
            self._client.post(
                "/interrupt",
                json={"prompt_id": prompt_id},
                timeout=2,
            )
        except httpx.HTTPError:
            LOGGER.warning("[IMAGE_EDIT] unable to interrupt timed-out prompt")

    @staticmethod
    def _find_unique_marker(
        graph: dict[str, dict[str, Any]], marker: str
    ) -> tuple[str, dict[str, Any]]:
        matches = [
            (node_id, node)
            for node_id, node in graph.items()
            if isinstance(node.get("_meta"), dict)
            and node["_meta"].get("title") == marker
        ]
        if len(matches) != 1:
            raise AppError(
                code=ErrorCode.IMAGE_EDIT_FAILED,
                stage=PipelineStage.IMAGE_EDIT,
                user_message=f"ComfyUI API 工作流必须包含唯一的 {marker} 标记。",
                retriable=False,
                detail=f"matches={len(matches)}",
            )
        return matches[0]

    @classmethod
    def _find_optional_marker(
        cls, graph: dict[str, dict[str, Any]], marker: str
    ) -> tuple[str, dict[str, Any]] | None:
        matches = [
            (node_id, node)
            for node_id, node in graph.items()
            if isinstance(node.get("_meta"), dict)
            and node["_meta"].get("title") == marker
        ]
        if len(matches) > 1:
            cls._raise_workflow_error(
                f"ComfyUI API 工作流中的 {marker} 标记不唯一。",
                {"matches": len(matches)},
            )
        return matches[0] if matches else None

    @classmethod
    def _inject_seed(cls, node: dict[str, Any], seed: int) -> None:
        inputs = node["inputs"]
        seed_keys = [key for key in ("noise_seed", "seed") if key in inputs]
        if len(seed_keys) != 1:
            cls._raise_workflow_error(
                "DSB_SEED 节点必须包含唯一的 seed 或 noise_seed 输入。",
                {"keys": seed_keys},
            )
        inputs[seed_keys[0]] = seed

    @classmethod
    def _validated_remote_record(cls, record: dict[str, Any]) -> tuple[str, str, str]:
        name = record.get("name", record.get("filename"))
        subfolder = record.get("subfolder", "")
        remote_type = record.get("type")
        if not all(isinstance(value, str) for value in (name, subfolder, remote_type)):
            cls._raise_workflow_error("ComfyUI 图片路径字段不完整。", record)
        if not name or name != PurePosixPath(name).name or "\\" in name:
            cls._raise_workflow_error("ComfyUI 返回了不安全的图片文件名。", record)
        if "\\" in subfolder:
            cls._raise_workflow_error("ComfyUI 返回了不安全的图片子目录。", record)
        folder_path = PurePosixPath(subfolder)
        if folder_path.is_absolute() or ".." in folder_path.parts:
            cls._raise_workflow_error("ComfyUI 返回了不安全的图片子目录。", record)
        return name, subfolder, remote_type

    @staticmethod
    def _prompt(appraisal: AppraisalResult, request: ScanRequest) -> str:
        # Klein has no prompt upsampling. Compile a short positive prompt from validated
        # scenario data instead of forwarding free-form feature nouns that can cause a
        # small model to redraw product geometry.
        intensity = ComfyUIProvider._weathering_intensity(request.apocalypse_years)
        effects = SCENARIO_SURFACE_EFFECTS[request.scenario.value]
        return (
            f"Apply {intensity} post-apocalyptic surface degradation to the photographed "
            "object. Make extreme, near-ruin, visually dominant surface degradation cover "
            "nearly every visible surface, "
            f"consistent with {request.apocalypse_years:g} years of "
            f"{request.scenario.value.lower().replace('_', ' ')}: {effects}. "
            "It remains the exact same individual object with its source-defined silhouette, "
            "proportions, and every structural detail in the original location. Its underlying "
            "color pattern remains recognizable. Keep its position, perspective, crop, lighting, "
            "shadows, and background unchanged. No missing parts, holes, warping, deformation, "
            "or new components. All functional geometry remains unchanged."
        )

    @staticmethod
    def _text_to_image_prompt(
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> str:
        item = request.description.strip() or appraisal.original_item
        materials = ", ".join(appraisal.materials[:4])
        features = ", ".join(appraisal.original_features[:6])
        changes = ", ".join(appraisal.apocalypse_changes[:5])
        effects = SCENARIO_SURFACE_EFFECTS[request.scenario.value]
        return (
            f"A photorealistic post-apocalyptic evidence photograph of exactly one {item}, "
            f"reimagined as {appraisal.apocalypse_name} after {request.apocalypse_years:g} "
            f"years of {request.scenario.value.lower().replace('_', ' ')}. "
            f"The object is made from {materials} and has these practical visible features: "
            f"{features}. Its evolved condition shows {changes}, with {effects}. "
            "The complete single object is centered and fully visible in a three-quarter view "
            "on a plain neutral gray evidence-table background, with realistic scale, coherent "
            "construction, soft documentary lighting, a natural shadow, and high material detail."
        )

    @staticmethod
    def _weathering_intensity(years: float) -> str:
        if years <= 0.25:
            return "strongly pronounced"
        if years <= 1:
            return "severe"
        if years <= 3:
            return "extreme, near-ruin"
        if years <= 10:
            return "devastating, long-neglected"
        return "ruin-grade, barely salvageable"

    @classmethod
    def _raise_workflow_error(
        cls,
        message: str,
        detail: Any,
        retriable: bool = False,
    ) -> None:
        raise AppError(
            code=ErrorCode.IMAGE_EDIT_FAILED,
            stage=PipelineStage.IMAGE_EDIT,
            user_message=message,
            retriable=retriable,
            detail=cls._detail(detail),
        )

    @classmethod
    def _raise_oom(cls, detail: Any) -> None:
        raise AppError(
            code=ErrorCode.GPU_OOM,
            stage=PipelineStage.IMAGE_EDIT,
            user_message="显存不足，已保留鉴定卡；释放模型后可单独重试 AFTER 图片。",
            retriable=True,
            detail=cls._detail(detail),
        )

    @staticmethod
    def _contains_oom(detail: str) -> bool:
        lowered = detail.lower()
        return any(marker in lowered for marker in OOM_MARKERS)

    @staticmethod
    def _detail(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)[:2000]
        except (TypeError, ValueError):
            return repr(value)[:2000]
