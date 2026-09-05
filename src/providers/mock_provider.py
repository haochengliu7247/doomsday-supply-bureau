from __future__ import annotations

import hashlib
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from src.schemas import (
    AppraisalResult,
    EffectLevel,
    EvidenceItem,
    Grade,
    ImageIdentityVerdict,
    ItemCategory,
    ItemStats,
    PlayerEffectProfile,
    RecommendedAction,
    ScanRequest,
)

SCENARIO_DAMAGE = {
    "CITY_BLACKOUT": ["厚重污垢", "密集刮擦", "大面积涂层脱落", "严重边缘磨损"],
    "NUCLEAR_RUINS": ["厚重灰烬结壳", "大面积涂层粉化", "密集冲击伤痕"],
    "FLOOD": ["厚重泥沙结壳", "多重深色水位线", "大面积潮湿锈蚀"],
    "EXTREME_COLD": ["密集霜蚀", "大面积非结构性冻裂", "严重边缘风化"],
    "EXTREME_HEAT": ["厚重烟炱", "大面积漆面起泡炭化", "严重焦灼磨损"],
    "NATURE_RECLAIMS": ["大片地衣结壳", "浓重苔藓斑", "大面积生物锈蚀层"],
}

SCENARIO_DAMAGE_EN = {
    "CITY_BLACKOUT": "thick embedded grime, dense abrasion, broad coating loss, and oxidation",
    "NUCLEAR_RUINS": "heavy ash crust, chalked coating, dense scars, and surface pitting",
    "FLOOD": "thick silt crust, dark waterlines, broad damp staining, and oxidation",
    "EXTREME_COLD": "dense frost etching, extensive cold crazing, and severe edge weathering",
    "EXTREME_HEAT": "heavy soot crust, blistered charred coating, and scorched edge wear",
    "NATURE_RECLAIMS": "established lichen crust, broad moss staining, and biological patina",
}


class MockVLMProvider:
    name = "mock-vlm"

    def analyze(
        self,
        image_path: Path | None,
        request: ScanRequest,
    ) -> AppraisalResult:
        item_name = request.description.strip() or "便携式充电宝"
        damage = SCENARIO_DAMAGE[request.scenario.value]
        year_text = (
            "三个月"
            if request.apocalypse_years == 0.25
            else f"{request.apocalypse_years:g}年"
        )
        return AppraisalResult(
            original_item=item_name,
            object_description="矩形便携设备，外壳完整，具有固定接口和状态按键。",
            observed_evidence=["单一矩形主体", "深色保护外壳", "前端存在固定接口"],
            material_evidence=[
                EvidenceItem(
                    observation="外壳呈细哑光纹理",
                    inference="工程塑料或表面涂层",
                    confidence=0.78,
                )
            ],
            materials=["工程塑料", "金属电子元件"],
            dominant_colors=["石墨黑"],
            original_features=["圆角矩形轮廓", "前端接口", "圆形状态按键"],
            preserve_features=[
                "保持原物轮廓与长宽比例",
                "保持拍摄视角和构图",
                "保持接口、按键数量与位置",
                "保持石墨黑基础配色",
            ],
            apocalypse_changes=[
                f"{year_text}{request.scenario.label}环境造成的{effect}"
                for effect in damage
            ],
            avoid_changes=[
                "不得改变物品类别",
                "不得添加武器或科幻组件",
                "不得改变接口和按键数量",
                "不得替换背景和拍摄角度",
            ],
            apocalypse_name="便携式能源储备核心",
            category=ItemCategory.ENERGY,
            stats=ItemStats(
                survival=4,
                scarcity=4,
                trade=5,
                storage=4,
                versatility=4,
                risk=1,
            ),
            grade=Grade.A,
            market_value_liters=12,
            market_value_description="可交换约 12L 净水，实际价值随电网状态变化。",
            hidden_use="拆解后可取得电芯、导线与小型接口组件。",
            analysis=(
                "在基础设施失效后，仍可工作的便携电源能维持照明、离线资料设备"
                "与短距离通信，是稳定而容易理解的高价值物资。"
            ),
            verdict="文明崩溃以后，电量百分比可能比银行卡余额更值得炫耀。",
            recommended_action=RecommendedAction.CARRY_NOW,
            player_effect_profile=PlayerEffectProfile(
                health=EffectLevel.NONE,
                energy=EffectLevel.SMALL_POSITIVE,
                hunger=EffectLevel.NONE,
                morale=EffectLevel.TINY_POSITIVE,
            ),
            image_edit_prompt=(
                "Apply severe, widespread, visually dominant post-apocalyptic degradation "
                f"across most surfaces: {SCENARIO_DAMAGE_EN[request.scenario.value]}, after "
                f"{request.apocalypse_years:g} years in a {request.scenario.value} scenario. "
                "Keep the exact same individual object, rounded shell, proportions, openings, "
                "control, underlying color pattern, position, perspective, crop, shadows, and "
                "background. No missing parts, holes, deformation, or added components."
            ),
            confidence=0.92,
            warnings=["当前为 Mock Mode：鉴定内容用于界面和规则验证。"],
        )

    def unload(self) -> None:
        return None

    def verify_identity(
        self,
        source_path: Path,
        candidate_path: Path,
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> ImageIdentityVerdict:
        return ImageIdentityVerdict(
            same_physical_object=True,
            category_preserved=True,
            silhouette_and_proportions_preserved=True,
            camera_and_composition_preserved=True,
            functional_features_preserved=True,
            only_surface_condition_changed=True,
            post_apocalyptic_damage_clearly_visible=True,
            before_functional_features=appraisal.original_features,
            after_functional_features=appraisal.original_features,
            added_features=[],
            missing_features=[],
            moved_or_duplicated_features=[],
            issues=[],
            confidence=1.0,
        )


class MockImageEditProvider:
    name = "mock-image-editor"

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def edit(
        self,
        image_path: Path | None,
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> Path | None:
        if image_path is None:
            return self._generate_from_text(appraisal, request)

        with Image.open(image_path) as source:
            base = source.convert("RGB")

        seed_material = f"{image_path.name}:{request.scenario.value}:{request.apocalypse_years}"
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)

        weathered = ImageEnhance.Color(base).enhance(0.72)
        weathered = ImageEnhance.Contrast(weathered).enhance(1.12)
        overlay = Image.new("RGBA", weathered.size, (60, 45, 24, 0))
        overlay_draw = ImageDraw.Draw(overlay, "RGBA")

        width, height = weathered.size
        severity = min(1.0, max(0.15, request.apocalypse_years / 12))
        spot_count = max(18, int((width * height / 45000) * (1 + severity)))
        for _ in range(spot_count):
            radius = rng.randint(max(2, width // 180), max(4, width // 55))
            x = rng.randint(0, width)
            y = rng.randint(0, height)
            alpha = rng.randint(8, 25)
            overlay_draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(82, 61, 35, alpha),
            )

        scratch_count = max(8, int(14 * severity))
        for _ in range(scratch_count):
            x1 = rng.randint(0, width)
            y1 = rng.randint(0, height)
            length = rng.randint(max(8, width // 40), max(16, width // 9))
            overlay_draw.line(
                (x1, y1, min(width, x1 + length), min(height, y1 + rng.randint(-8, 8))),
                fill=(220, 210, 190, rng.randint(25, 60)),
                width=max(1, width // 700),
            )

        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(0.3, width / 1800)))
        result = Image.alpha_composite(weathered.convert("RGBA"), overlay).convert("RGB")
        output_path = self.output_dir / f"{image_path.stem}_after.jpg"
        result.save(output_path, quality=92, optimize=True)
        return output_path

    def _generate_from_text(
        self,
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> Path:
        seed_material = (
            f"{request.description}:{request.scenario.value}:"
            f"{request.apocalypse_years}:{appraisal.apocalypse_name}"
        )
        digest = hashlib.sha256(seed_material.encode()).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        canvas = Image.new("RGB", (768, 768), (200, 197, 185))
        draw = ImageDraw.Draw(canvas)
        body_color = (
            rng.randint(45, 85),
            rng.randint(48, 82),
            rng.randint(42, 70),
        )
        draw.rounded_rectangle(
            (155, 170, 613, 610),
            radius=70,
            fill=body_color,
            outline=(32, 31, 27),
            width=8,
        )
        for _ in range(28):
            x = rng.randint(180, 585)
            y = rng.randint(195, 580)
            length = rng.randint(18, 85)
            draw.line(
                (x, y, min(600, x + length), y + rng.randint(-6, 6)),
                fill=(155, 143, 112),
                width=2,
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"dsb_text_{digest[:16]}_after.jpg"
        canvas.save(output_path, quality=92, optimize=True)
        return output_path

    def unload(self) -> None:
        return None
