() => {
    "use strict";

    // This layer only presents the scan result. It never submits a request or edits state.
    if (window.dsbPackReveal) return;
    let active = null;
    const colors = {
        S: "#ffd36a", A: "#ffab63", B: "#bd99ff", C: "#82c9ff",
        D: "#9ed9a3", E: "#ced1c7", F: "#b8b6ad",
    };
    const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function later(run, callback, delay) {
        const id = window.setTimeout(() => {
            run.timers.delete(id);
            if (active === run) callback();
        }, delay);
        run.timers.add(id);
    }

    function close() {
        const run = active;
        if (!run) return;
        active = null;
        for (const timer of run.timers) window.clearTimeout(timer);
        run.timers.clear();
        if (run.dialog.open) run.dialog.close();
        run.dialog.remove();
        if (run.previousFocus?.isConnected) run.previousFocus.focus({preventScroll: true});
    }

    function safeImageUrl(value) {
        const candidate = typeof value === "string" ? value : (value?.url || value?.image?.url);
        if (!candidate) return null;
        try {
            const url = new URL(candidate, window.location.href);
            if (url.origin === window.location.origin && /^https?:$/.test(url.protocol)) {
                return url.href;
            }
        } catch (_) { /* An absent preview must not prevent the result from being shown. */ }
        return null;
    }

    function start() {
        if (active) return;
        const dialog = document.createElement("dialog");
        dialog.id = "dsb-pack-reveal";
        dialog.className = "dsb-pack is-charging";
        dialog.setAttribute("aria-labelledby", "dsb-pack-title");
        dialog.setAttribute("aria-describedby", "dsb-pack-status");
        // Static UI only; all model-generated strings are assigned with textContent below.
        dialog.innerHTML = `
            <div class="pack-vignette" aria-hidden="true"></div>
            <div class="pack-topline"><span>DSB / 封存物资</span>
                <button type="button" class="pack-skip" autofocus>跳过动画 <span>×</span></button>
            </div>
            <div class="pack-heading">
                <p>WASTELAND ARCHIVE</p>
                <h2 id="dsb-pack-title">正在唤醒旧世界的遗物</h2>
            </div>
            <div class="pack-stage">
                <div class="pack-rays" aria-hidden="true"></div>
                <div class="pack-orbit orbit-one" aria-hidden="true"></div>
                <div class="pack-orbit orbit-two" aria-hidden="true"></div>
                <div class="pack-shockwave" aria-hidden="true"></div>
                <div class="pack-particles" aria-hidden="true"></div>
                <div class="pack-core"><div class="pack-float"><div class="pack-card">
                    <div class="pack-face pack-back" aria-hidden="true">
                        <span class="pack-serial">DOOMSDAY SUPPLY BUREAU</span>
                        <div class="pack-insignia"><span>DSB</span><b>?</b><small>未知物资</small></div>
                        <div class="pack-seal">封 存 · 待 鉴</div>
                        <span class="pack-back-footer">从废墟中，重新发现价值。</span>
                    </div>
                    <div class="pack-face pack-front" aria-hidden="true">
                        <div class="pack-front-top"><span>鉴定档案</span><b class="pack-grade"></b></div>
                        <div class="pack-image-frame"><img class="pack-result-image" alt="" hidden>
                            <span class="pack-image-fallback">物资档案</span></div>
                        <div class="pack-result-copy"><p class="pack-grade-label"></p>
                            <h3 class="pack-result-name"></h3><p class="pack-original-name"></p></div>
                    </div>
                </div></div></div>
                <div class="pack-flare" aria-hidden="true"></div>
            </div>
            <div class="pack-bottom">
                <p id="dsb-pack-status" role="status" aria-live="polite">鉴定中 · 正在揭开它的灾后身份</p>
                <button type="button" class="pack-continue" hidden>查看鉴定结果 <span>→</span></button>
                <span class="pack-wait-detail">鉴定完成后自动揭晓</span>
            </div>`;
        const run = {
            dialog, timers: new Set(), started: performance.now(),
            previousFocus: document.activeElement, finishing: false,
        };
        active = run;
        const particles = dialog.querySelector(".pack-particles");
        for (let i = 0; i < 48; i++) {
            const particle = document.createElement("i");
            const angle = i * 2.39996;
            const distance = 160 + (i % 9) * 26;
            particle.style.setProperty("--spark-x", `${Math.cos(angle) * distance}px`);
            particle.style.setProperty("--spark-y", `${Math.sin(angle) * distance}px`);
            particle.style.setProperty("--spark-delay", `${(i % 7) * 23}ms`);
            particle.style.setProperty("--spark-size", `${2 + (i % 4)}px`);
            particles.appendChild(particle);
        }
        dialog.querySelector(".pack-skip").addEventListener("click", close);
        dialog.querySelector(".pack-continue").addEventListener("click", close);
        dialog.addEventListener("cancel", event => { event.preventDefault(); close(); });
        document.body.appendChild(dialog);
        try { dialog.showModal(); } catch (_) { close(); return; }
        later(run, () => {
            dialog.querySelector(".pack-wait-detail").textContent = "物资仍在鉴定中，可跳过动画查看进度";
        }, 30000);
        // A disconnected request must never leave an undismissable layer behind.
        later(run, close, 15 * 60 * 1000);
    }

    function finish(result, image) {
        const run = active;
        if (!run || run.finishing) return;
        if (!result || !["success", "partial"].includes(result.status)) {
            close();
            return;
        }
        run.finishing = true;
        const {dialog} = run;
        const grade = Object.hasOwn(colors, result.grade) ? result.grade : "F";
        dialog.style.setProperty("--pack-color", colors[grade]);
        dialog.dataset.grade = grade;
        dialog.querySelector(".pack-grade").textContent = grade;
        dialog.querySelector(".pack-grade-label").textContent = `${grade} 级物资`;
        dialog.querySelector(".pack-result-name").textContent = result.name || "物资档案";
        dialog.querySelector(".pack-original-name").textContent = result.original_name || "";
        dialog.querySelector(".pack-image-fallback").textContent =
            result.status === "partial" ? "图片待补充" : "鉴定档案已签发";
        const picture = dialog.querySelector(".pack-result-image");
        const imageUrl = safeImageUrl(image);
        if (imageUrl && result.status === "success") {
            picture.onload = () => {
                if (active !== run) return;
                picture.hidden = false;
                dialog.querySelector(".pack-image-fallback").hidden = true;
            };
            picture.onerror = () => { picture.hidden = true; };
            picture.alt = `${result.name || "物资"}的灾后形态`;
            picture.src = imageUrl;
        }
        // Even an immediate cache hit gets a brief opening, without delaying the scan itself.
        const delay = reducedMotion() ? 0 : Math.max(0, 1500 - (performance.now() - run.started));
        later(run, () => {
            dialog.classList.replace("is-charging", "is-opening");
            dialog.querySelector("#dsb-pack-title").textContent = "封存解除";
            dialog.querySelector("#dsb-pack-status").textContent = "鉴定完成 · 遗物正在显现";
            dialog.querySelector(".pack-wait-detail").hidden = true;
            later(run, () => {
                dialog.classList.add("is-revealed");
                dialog.querySelector(".pack-front").setAttribute("aria-hidden", "false");
                dialog.querySelector("#dsb-pack-title").textContent = grade === "S" ? "非凡遗物，重见天日" : "旧世界的遗物，新的身份";
                dialog.querySelector("#dsb-pack-status").textContent = result.status === "partial"
                    ? "鉴定已完成 · 图片可在页面单独重试" : `${grade} 级物资 · ${result.name || "鉴定完成"}`;
                const next = dialog.querySelector(".pack-continue");
                next.hidden = false;
                dialog.querySelector(".pack-skip").textContent = "关闭 ×";
                next.focus({preventScroll: true});
            }, reducedMotion() ? 0 : 1150);
        }, delay);
    }

    window.dsbPackReveal = {start, finish, abort: close};
    window.addEventListener("pagehide", close);
}
