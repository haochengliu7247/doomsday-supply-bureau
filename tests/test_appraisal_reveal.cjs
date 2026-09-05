// Frontend state tests with a minimal DOM adapter; no server, browser or player save.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const test = require("node:test");
const path = require("node:path");
const source = fs.readFileSync(path.join(__dirname, "../assets/appraisal_reveal.js"), "utf8");

function harness({reduce = false} = {}) {
    let now = 0, serial = 0;
    const timers = new Map(), dialogs = [];
    class Element {
        constructor() {
            this.nodes = new Map(); this.handlers = {}; this.dataset = {};
            this.style = {setProperty() {}}; this.classes = new Set();
            this.classList = {
                add: name => this.classes.add(name),
                replace: (old, name) => { this.classes.delete(old); this.classes.add(name); },
            };
        }
        setAttribute(name, value) { this[name] = value; }
        querySelector(selector) {
            if (!this.nodes.has(selector)) this.nodes.set(selector, new Element());
            return this.nodes.get(selector);
        }
        appendChild() {}
        addEventListener(name, callback) { this.handlers[name] = callback; }
        showModal() { this.open = true; }
        close() { this.open = false; }
        remove() { this.removed = true; }
        focus() { this.focused = true; }
    }
    const window = {
        location: {href: "http://127.0.0.1:7861/", origin: "http://127.0.0.1:7861"},
        matchMedia: () => ({matches: reduce}), addEventListener() {},
        setTimeout(callback, delay) { const id = ++serial; timers.set(id, {at: now + delay, callback}); return id; },
        clearTimeout(id) { timers.delete(id); },
    };
    const document = {
        activeElement: {isConnected: true, focus() {}},
        body: {appendChild: dialog => dialogs.push(dialog)},
        createElement: () => new Element(),
    };
    vm.runInNewContext(`(${source})()`, {window, document, URL, performance: {now: () => now}});
    function advance(ms) {
        const until = now + ms;
        while (true) {
            const entry = [...timers].filter(([, timer]) => timer.at <= until)
                .sort((a, b) => a[1].at - b[1].at)[0];
            if (!entry) break;
            now = entry[1].at; timers.delete(entry[0]); entry[1].callback();
        }
        now = until;
    }
    return {api: window.dsbPackReveal, dialogs, timers, advance};
}
const result = {status: "success", name: "余烬水囊", original_name: "水瓶", grade: "S"};

test("instant cache result waits for opening, then reveals actual grade and image", () => {
    const h = harness(); h.api.start(); h.api.start();
    assert.equal(h.dialogs.length, 1);
    h.api.finish(result, {url: "/gradio_api/file=after.png"});
    const d = h.dialogs[0];
    h.advance(1499); assert.equal(d.classes.has("is-opening"), false);
    h.advance(1); assert.equal(d.classes.has("is-opening"), true);
    h.advance(1150); assert.equal(d.classes.has("is-revealed"), true);
    assert.equal(d.querySelector(".pack-result-name").textContent, result.name);
    assert.equal(d.dataset.grade, "S");
    assert.equal(d.querySelector(".pack-result-image").src, "http://127.0.0.1:7861/gradio_api/file=after.png");
});
test("slow generation stays charging until result, without inventing a grade", () => {
    const h = harness(); h.api.start(); h.advance(45000);
    const d = h.dialogs[0];
    assert.equal(d.classes.has("is-revealed"), false); assert.equal(d.dataset.grade, undefined);
    h.api.finish(result, null); h.advance(1150);
    assert.equal(d.classes.has("is-revealed"), true);
});
test("skip cancels only animation and late completion cannot reopen it", () => {
    const h = harness(); h.api.start(); const d = h.dialogs[0];
    d.querySelector(".pack-skip").handlers.click();
    h.api.finish(result, null); h.advance(10000);
    assert.equal(d.removed, true); assert.equal(h.timers.size, 0);
    assert.equal(h.dialogs.length, 1); assert.equal(d.classes.has("is-revealed"), false);
});
test("failure clears the modal instead of revealing any old card", () => {
    const h = harness(); h.api.start(); h.api.finish({status: "error"}, {url: "/old.png"});
    assert.equal(h.dialogs[0].removed, true); assert.equal(h.timers.size, 0);
    h.api.start(); h.api.abort(); assert.equal(h.dialogs[1].removed, true);
});
test("a partial result never shows the old image and reduced motion reveals immediately", () => {
    const h = harness({reduce: true}); h.api.start();
    h.api.finish({...result, status: "partial"}, {url: "/old.png"}); h.advance(0);
    const d = h.dialogs[0];
    assert.equal(d.classes.has("is-revealed"), true);
    assert.equal(d.querySelector(".pack-result-image").src, undefined);
    assert.equal(d.querySelector(".pack-image-fallback").textContent, "图片待补充");
});
test("model text stays text, and remote image URLs are not loaded", () => {
    const h = harness({reduce: true}); h.api.start();
    const name = '<img src=x onerror="bad()">';
    h.api.finish({...result, name}, {url: "https://example.com/image.png"}); h.advance(0);
    assert.equal(h.dialogs[0].querySelector(".pack-result-name").textContent, name);
    assert.equal(h.dialogs[0].querySelector(".pack-result-image").src, undefined);
});
