import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const dashboardDir = new URL("../pages/dashboard/", import.meta.url);
const appSource = readFileSync(new URL("app.js", dashboardDir), "utf8");
const stylesSource = readFileSync(new URL("styles.css", dashboardDir), "utf8");
const indexSource = readFileSync(new URL("index.html", dashboardDir), "utf8");
const elementIds = [
  "workspace",
  "empty",
  "session-id",
  "load",
  "theme",
  "mood-label",
  "hero-h2",
  "hero-desc",
  "core-hint",
  "event-count",
  "ev-total",
  "events",
  "action-status",
  "event-orbits",
  "attention-count",
  "attention-items",
  "intimacy-stage",
  "intimacy-tier",
  "intimacy",
  "diary-count",
  "diaries",
  "diary-detail",
  "diary-search",
  "diary-month",
  "diary-prev",
  "diary-next",
  "diary-page",
  "diagnostics",
  "refresh-prompt",
  "prompt-kind",
  "prompt",
  "preview-kind",
  "preview-prompt",
  "rule-sample",
  "run-rule",
  "rule-output",
  "rain",
  "delete-confirm",
  "delete-confirm-message",
  "delete-cancel",
  "delete-apply",
];

function createElement(id) {
  const classes = new Set();
  const listeners = new Map();
  return {
    id,
    hidden: id === "workspace" || id === "delete-confirm",
    disabled: false,
    value: "",
    innerHTML: "",
    textContent: "",
    dataset: {},
    style: {},
    classList: {
      remove: (...names) => names.forEach((name) => classes.delete(name)),
      toggle: (name, enabled) => enabled ? classes.add(name) : classes.delete(name),
      contains: (name) => classes.has(name),
    },
    addEventListener: (type, listener) => listeners.set(type, listener),
    dispatch: (type, event) => listeners.get(type)?.(event),
    focus: () => {},
    appendChild: () => {},
  };
}

function diary(day, month = "03") {
  const cycleDate = `2026-${month}-${String(day).padStart(2, "0")}`;
  return {
    cycle_date: cycleDate,
    diary: `${cycleDate} 的日记正文`,
    day_summary: `${cycleDate} 的回顾摘要`,
    provider_id: "test",
    mood_proposal: { valence: 0.2, energy: 0.5, tension: 0.1 },
  };
}

function ledger(diaries) {
  return {
    state_version: 9,
    mood: { label: "平静", valence: 0, energy: 0.5, tension: 0.1 },
    events: [
      {
        id: 'event-1"unsafe',
        version: 4,
        fact: "他准备去上班了",
        emotional_meaning: "这件日常小事让我惦记",
        category: "episodic",
        lifecycle: "active",
      },
    ],
    attention_items: [
      { id: "attention-1<unsafe", status: "open" },
    ],
    intimacy: {
      stage: "not_noticeable",
      body_sensitivity: 1,
      sexual_arousal: 0.89,
      intimacy_willingness: 0.15,
      inhibition: 0.85,
    },
    diaries,
  };
}

async function settle() {
  for (let index = 0; index < 8; index += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

function run(context, source) {
  return vm.runInContext(source, context);
}

test("long event rows keep their status controls inside the panel", () => {
  assert.match(
    stylesSource,
    /\.event-item\s*\{[^}]*grid-template-columns:\s*8px\s+minmax\(0,\s*1fr\)\s+auto/s,
  );
  assert.match(stylesSource, /\.event-item\s*>\s*div\s*\{[^}]*min-width:\s*0/s);
  assert.match(stylesSource, /\.event-item strong\s*\{[^}]*overflow-wrap:\s*anywhere/s);
  assert.match(stylesSource, /\.event-state\s*\{[^}]*min-width:\s*86px/s);
  assert.match(stylesSource, /\.delete-item\s*\{[^}]*white-space:\s*nowrap/s);
});

test("emotion injection views are split into dedicated half-width panels", () => {
  assert.match(indexSource, /情绪注入 · 历史快照/);
  assert.match(indexSource, /情绪注入 · 实时预览/);
  assert.doesNotMatch(indexSource, /injection-grid|injection-view/);
  assert.doesNotMatch(stylesSource, /injection-grid|injection-view/);
  assert.match(stylesSource, /\.adv-body\s*\{[^}]*grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)/s);
  for (const id of ["refresh-prompt", "prompt-kind", "prompt", "preview-kind", "preview-prompt"]) {
    const matches = indexSource.match(new RegExp(`id="${id}"`, "g")) || [];
    assert.equal(matches.length, 1, `id "${id}" 应在 index.html 中恰好出现一次`);
  }
});

test("brand mark shows the synced plugin logo with a text fallback", () => {
  assert.match(indexSource, /class="mark-logo"\s+src="\.\/logo\.png"/);
  assert.match(indexSource, /class="mark-fallback"/);
  assert.match(stylesSource, /\.mark\s*\{[^}]*overflow:\s*hidden/s);
  assert.match(stylesSource, /\.mark \.mark-logo\s*\{[^}]*object-fit:\s*cover/s);
  assert.match(stylesSource, /\.mark \.mark-logo\s*\{[^}]*border-radius:\s*50%/s);
});

test("heartbeat glow anchors to the mood stage, not the hero card", () => {
  const stageIdx = indexSource.indexOf('<div class="stage"');
  const glowIdx = indexSource.indexOf('class="hero-glow"');
  const ringIdx = indexSource.indexOf('class="ring r1"');
  assert.ok(stageIdx !== -1, "stage 容器必须存在");
  assert.ok(glowIdx > stageIdx && glowIdx < ringIdx, "hero-glow 应是 stage 的第一个子元素");
  assert.match(
    stylesSource,
    /\.hero-glow\s*\{[^}]*animation:glowbeat\s+var\(--hb-dur\)\s+ease-in-out infinite\}/s,
    "hero-glow 只应保留 glowbeat 动画",
  );
});

test("daily review browser handles loading, filtering, and pagination", async () => {
  const elements = Object.fromEntries(elementIds.map((id) => [id, createElement(id)]));
  const payload = {
    ledger: ledger([diary(1)]),
    presentation: {
      persona_intimacy_tier: "对你有很强的身体吸引",
      body_reaction_stage: "身体反应强烈，敏感度和性唤起处于高位",
      attention_items: [
        {
          item_id: "attention-1<unsafe",
          item_version: 3,
          content: "等一下咱们来玩角色扮演",
          kind: "plan",
          kind_label: "计划",
          status: "open",
          status_label: "仍待关注",
          time_hint: "等一下",
          overdue: false,
        },
      ],
    },
    diagnostics: {},
  };
  const postCalls = [];
  let stateGetCount = 0;
  let deleteFailure = "";
  const bridge = {
    ready: async () => {},
    apiGet: async (path) => {
      if (path === "page/sessions") {
        return { status: "ok", data: { sessions: [{ session_id: "private:test" }] } };
      }
      if (path === "page/state") {
        stateGetCount += 1;
        return { status: "ok", data: payload };
      }
      if (path === "page/injection") {
        return {
          status: "ok",
          data: {
            kind: "actual",
            prompt: "actual",
            current_state_version: 9,
            actual: {
              prompt: "actual",
              state_version: 7,
              generated_at: "2026-08-04T13:41:28+00:00",
              request_source: "spark_proactive",
              marker_complete: true,
              stale: true,
            },
            preview: {
              prompt: "preview",
              state_version: 9,
              generated_at: "2026-08-04T13:45:00+00:00",
              request_source: "preview",
              marker_complete: true,
              stale: false,
            },
          },
        };
      }
      throw new Error(`Unexpected GET ${path}`);
    },
    apiPost: async (path, body) => {
      postCalls.push({ path, body });
      if (path === "page/items/delete" && deleteFailure) {
        return { status: "error", message: deleteFailure };
      }
      return { status: "ok", data: path === "page/rules/test" ? { run: {} } : {} };
    },
  };
  const windowObject = {
    AstrBotPluginPage: bridge,
    localStorage: { getItem: () => "", setItem: () => {} },
    setTimeout: (callback, milliseconds) => {
      if (milliseconds >= 1000) return 0;
      return setTimeout(callback, milliseconds);
    },
  };
  const context = vm.createContext({
    console,
    document: {
      getElementById: (id) => elements[id],
      documentElement: { dataset: {} },
      createElement: () => ({ style: {}, appendChild: () => {} }),
    },
    setImmediate,
    setTimeout,
    window: windowObject,
  });

  vm.runInContext(appSource, context, { filename: "app.js" });
  await settle();

  assert.equal(elements.empty.hidden, true);
  assert.equal(elements.workspace.hidden, false);
  assert.equal(elements["diary-count"].textContent, "1 / 1 篇");
  assert.match(elements.events.innerHTML, /近期片段/);
  assert.match(elements.events.innerHTML, /他准备去上班了/);
  assert.match(elements.events.innerHTML, /data-delete-kind="event"/);
  assert.match(elements.events.innerHTML, /data-delete-id="event-1&quot;unsafe"/);
  assert.equal(
    elements["intimacy-stage"].textContent,
    "身体反应强烈，敏感度和性唤起处于高位",
  );
  assert.equal(
    elements["intimacy-tier"].textContent,
    "对你有很强的身体吸引",
  );
  assert.equal(
    run(context, "intimacyStageLabel('open_and_receptive')"),
    "身体反应强烈，敏感度和性唤起处于高位",
  );
  assert.equal(run(context, "intimacyStageLabel('legacy_unknown')"), "未知身体反应");
  assert.doesNotMatch(elements.intimacy.innerHTML, /亲近意愿|克制程度/);
  assert.match(elements.intimacy.innerHTML, /1\.00/);
  assert.match(elements.intimacy.innerHTML, /0\.89/);
  assert.match(elements["attention-items"].innerHTML, /等一下咱们来玩角色扮演/);
  assert.match(elements["attention-items"].innerHTML, /仍待关注/);
  assert.match(elements["attention-items"].innerHTML, /data-delete-kind="attention"/);
  assert.match(elements["attention-items"].innerHTML, /data-delete-id="attention-1&lt;unsafe"/);
  assert.doesNotMatch(elements["intimacy-stage"].textContent, /not_noticeable/);
  assert.match(elements["diary-detail"].innerHTML, /2026-03-01 的日记正文/);
  assert.equal(elements.prompt.textContent, "actual");
  assert.match(elements["prompt-kind"].textContent, /Spark 主动对话/);
  assert.match(
    elements["prompt-kind"].textContent,
    /历史请求时点，已落后于当前实时账本 v9/,
  );
  assert.match(elements["prompt-kind"].textContent, /请求装配于 2026-08-04 21:41:28/);
  assert.match(elements["prompt-kind"].textContent, /北京时间 UTC\+08:00/);
  assert.doesNotMatch(elements["prompt-kind"].textContent, /2026-08-04T13:41:28\+00:00/);
  assert.equal(elements["preview-prompt"].textContent, "preview");
  assert.match(elements["preview-kind"].textContent, /与当前实时账本版本一致/);
  assert.match(elements["preview-kind"].textContent, /预览生成于/);

  const eventRow = {
    removed: false,
    remove() {
      this.removed = true;
    },
  };
  const eventButton = {
    dataset: {
      deleteKind: "event",
      deleteId: 'event-1"unsafe',
    },
    closest: (selector) => selector === ".event-item" ? eventRow : eventButton,
  };
  elements.events.dispatch("click", {
    target: { closest: () => eventButton },
  });
  assert.equal(postCalls.length, 0);
  assert.equal(elements["delete-confirm"].hidden, false);
  assert.match(elements["delete-confirm-message"].textContent, /删除这条心事/);

  elements["delete-cancel"].dispatch("click", {});
  assert.equal(elements["delete-confirm"].hidden, true);
  assert.equal(postCalls.length, 0);

  elements.events.dispatch("click", {
    target: { closest: () => eventButton },
  });
  const stateGetsBeforeDelete = stateGetCount;
  elements["delete-apply"].dispatch("click", {});
  await settle();
  assert.equal(elements["delete-confirm"].hidden, true);
  assert.equal(postCalls[0].path, "page/items/delete");
  assert.deepEqual(JSON.parse(JSON.stringify(postCalls[0].body)), {
    session_id: "private:test",
    kind: "event",
    id: 'event-1"unsafe',
  });
  assert.equal(stateGetCount, stateGetsBeforeDelete);
  assert.equal(eventRow.removed, true);
  assert.equal(elements["event-count"].textContent, "0 条");
  assert.equal(elements["action-status"].textContent, "心事已删除");
  assert.equal(elements["action-status"].hidden, false);

  const attentionRow = {
    removed: false,
    remove() {
      this.removed = true;
    },
  };
  const attentionButton = {
    dataset: {
      deleteKind: "attention",
      deleteId: "attention-1<unsafe",
    },
    closest: (selector) => selector === ".event-item" ? attentionRow : attentionButton,
  };
  elements["attention-items"].dispatch("click", {
    target: { closest: () => attentionButton },
  });
  assert.equal(postCalls.length, 1);
  assert.equal(elements["delete-confirm"].hidden, false);
  assert.match(elements["delete-confirm-message"].textContent, /删除这条待关注事项/);
  elements["delete-apply"].dispatch("click", {});
  await settle();
  assert.equal(postCalls[1].path, "page/items/delete");
  assert.deepEqual(JSON.parse(JSON.stringify(postCalls[1].body)), {
    session_id: "private:test",
    kind: "attention",
    id: "attention-1<unsafe",
  });
  assert.equal(stateGetCount, stateGetsBeforeDelete);
  assert.equal(attentionRow.removed, true);
  assert.equal(elements["attention-count"].textContent, "0 条");
  assert.equal(elements["action-status"].textContent, "待关注事项已删除");

  deleteFailure = "对象不存在，可能已被清理";
  const failedAttentionButton = {
    dataset: {
      deleteKind: "attention",
      deleteId: "attention-missing",
    },
    closest: (selector) => selector === ".event-item" ? attentionRow : failedAttentionButton,
  };
  elements["attention-items"].dispatch("click", {
    target: { closest: () => failedAttentionButton },
  });
  assert.equal(postCalls.length, 2);
  assert.equal(elements["delete-confirm"].hidden, false);
  elements["delete-apply"].dispatch("click", {});
  await settle();
  assert.equal(postCalls[2].path, "page/items/delete");
  assert.deepEqual(JSON.parse(JSON.stringify(postCalls[2].body)), {
    session_id: "private:test",
    kind: "attention",
    id: "attention-missing",
  });
  assert.match(elements["action-status"].textContent, /删除失败：对象不存在/);
  assert.equal(elements["action-status"].classList.contains("error"), true);
  assert.equal(stateGetCount, stateGetsBeforeDelete);

  context.__payload = { ledger: ledger([]) };
  run(context, "state.data = __payload; renderDiaries()");
  assert.equal(elements["diary-count"].textContent, "0 / 0 篇");
  assert.match(elements.diaries.innerHTML, /暂无每日回顾/);
  assert.match(elements["diary-detail"].innerHTML, /没有符合条件/);

  context.__payload = { ledger: ledger(Array.from({ length: 12 }, (_, index) => diary(index + 1))) };
  run(context, "state.data = __payload; state.diary = { page: 1, pageSize: 10, query: '', month: '', selectedDate: '' }; renderDiaries()");
  assert.equal(elements["diary-page"].textContent, "第 1 / 2 页");
  assert.equal((elements.diaries.innerHTML.match(/data-diary-date=/g) || []).length, 10);
  assert.match(elements.diaries.innerHTML, /2026-03-12/);
  assert.doesNotMatch(elements.diaries.innerHTML, /2026-03-01/);

  run(context, "state.diary.page = 2; state.diary.selectedDate = ''; renderDiaries()");
  assert.equal((elements.diaries.innerHTML.match(/data-diary-date=/g) || []).length, 2);
  assert.match(elements.diaries.innerHTML, /2026-03-01/);

  run(context, "state.diary.page = 1; state.diary.month = '2026-03'; state.diary.query = '不存在'; state.diary.selectedDate = ''; renderDiaries()");
  assert.equal(elements["diary-count"].textContent, "0 / 12 篇");
  assert.match(elements.diaries.innerHTML, /暂无每日回顾/);
  assert.match(elements["diary-detail"].innerHTML, /没有符合条件/);
});
