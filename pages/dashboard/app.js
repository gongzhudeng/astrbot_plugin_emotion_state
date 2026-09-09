let bridge = window.AstrBotPluginPage || null;
let bridgeReady = false;
let bridgeReadyPromise = null;
const state = {
  sessionId: readSessionId(),
  data: null,
  pendingDelete: null,
  diary: { page: 1, pageSize: 10, query: "", month: "", selectedDate: "" },
};
const $ = (id) => document.getElementById(id);
const wait = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

function readSessionId() {
  try {
    return window.localStorage?.getItem("emotion-state-session") || "";
  } catch {
    return "";
  }
}

function writeSessionId(sessionId) {
  try {
    window.localStorage?.setItem("emotion-state-session", sessionId);
  } catch {
    // Plugin pages may run in an opaque-origin iframe without storage access.
  }
}

function withTimeout(promise, milliseconds, message) {
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      window.setTimeout(() => reject(new Error(message)), milliseconds);
    }),
  ]);
}

function endpoint(path) {
  return String(path).replace(/^\/+/, "");
}

async function ensureBridgeReady() {
  for (let attempt = 0; attempt < 50 && !bridge; attempt += 1) {
    bridge = window.AstrBotPluginPage || null;
    if (!bridge) await wait(100);
  }
  if (!bridge) throw new Error("请从 AstrBot 官方插件页面打开此页面");
  if (!bridgeReady) {
    bridgeReadyPromise ||= withTimeout(
      bridge.ready(),
      10000,
      "插件页面桥接初始化超时，请刷新页面重试",
    );
    try {
      await bridgeReadyPromise;
      bridgeReady = true;
    } catch (error) {
      bridgeReadyPromise = null;
      throw error;
    }
  }
  return bridge;
}

async function api(path, options = {}) {
  const client = await ensureBridgeReady();
  const response = options.method === "POST"
    ? await client.apiPost(endpoint(path), options.body || {})
    : await client.apiGet(endpoint(path), options.params || {});
  if (response?.status === "error") throw new Error(response.message || "请求失败");
  return response?.data || response;
}

function showStatus(message, kind = "loading") {
  $("empty").hidden = false;
  $("workspace").hidden = true;
  $("empty").textContent = message;
  $("empty").classList.toggle("error", kind === "error");
}

function text(value) {
  return String(value ?? "");
}

function html(value) {
  return text(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}

function sessionLabel(session) {
  const parts = text(session.session_id).split(":");
  if (parts.length >= 3) return `${parts.at(-2)} · ${parts.at(-1)}`;
  return text(session.session_id);
}

function renderSessions(sessions) {
  const available = Array.isArray(sessions) ? sessions : [];
  const selected = available.some((item) => item.session_id === state.sessionId)
    ? state.sessionId
    : available[0]?.session_id || "";
  state.sessionId = selected;
  $("session-id").innerHTML = available.map((item) => `
    <option value="${html(item.session_id)}">${html(sessionLabel(item))}</option>
  `).join("");
  $("session-id").value = selected;
  $("session-id").disabled = available.length === 0;
  $("load").disabled = available.length === 0;
  if (selected) writeSessionId(selected);
  return Boolean(selected);
}

function diaryHaystack(item) {
  return [item.cycle_date, item.diary, item.day_summary, item.provider_id]
    .map(text)
    .join(" ")
    .toLocaleLowerCase();
}

function renderDiaryDetail(item) {
  if (!item) {
    $("diary-detail").innerHTML = `<p class="status-line">没有符合条件的每日回顾</p>`;
    return;
  }
  const proposal = item.mood_proposal || {};
  const moodValues = [
    ["偏向", proposal.valence],
    ["能量", proposal.energy],
    ["紧张", proposal.tension],
  ].filter(([, value]) => Number.isFinite(Number(value)));
  $("diary-detail").innerHTML = `
    <header><span class="diary-date">${html(item.cycle_date)}</span><small>${html(item.provider_id)}</small></header>
    <h3>当天日记</h3>
    <p class="diary-body">${html(item.diary)}</p>
    <h3>回顾摘要</h3>
    <p>${html(item.day_summary)}</p>
    ${moodValues.length ? `<div class="diary-mood">${moodValues.map(([label, value]) => `<span>${label} ${Number(value).toFixed(2)}</span>`).join("")}</div>` : ""}
  `;
}

function renderDiaries() {
  const diaries = Array.isArray(state.data?.ledger?.diaries)
    ? state.data.ledger.diaries.slice().reverse()
    : [];
  const months = [...new Set(diaries.map((item) => text(item.cycle_date).slice(0, 7)).filter(Boolean))];
  const currentMonth = state.diary.month;
  $("diary-month").innerHTML = [
    `<option value="">全部月份</option>`,
    ...months.map((month) => `<option value="${html(month)}">${html(month)}</option>`),
  ].join("");
  $("diary-month").value = months.includes(currentMonth) ? currentMonth : "";
  state.diary.month = $("diary-month").value;

  const query = state.diary.query.toLocaleLowerCase();
  const filtered = diaries.filter((item) => (
    (!state.diary.month || text(item.cycle_date).startsWith(state.diary.month))
    && (!query || diaryHaystack(item).includes(query))
  ));
  const totalPages = Math.max(1, Math.ceil(filtered.length / state.diary.pageSize));
  state.diary.page = Math.min(Math.max(1, state.diary.page), totalPages);
  const start = (state.diary.page - 1) * state.diary.pageSize;
  const pageItems = filtered.slice(start, start + state.diary.pageSize);
  if (!filtered.some((item) => item.cycle_date === state.diary.selectedDate)) {
    state.diary.selectedDate = pageItems[0]?.cycle_date || "";
  }

  $("diary-count").textContent = `${filtered.length} / ${diaries.length} 篇`;
  $("diaries").innerHTML = pageItems.map((item) => `
    <button class="diary-index${item.cycle_date === state.diary.selectedDate ? " selected" : ""}" type="button" data-diary-date="${html(item.cycle_date)}">
      <span>${html(item.cycle_date)}</span>
      <strong>${html(item.day_summary || item.diary || "无摘要")}</strong>
    </button>
  `).join("") || `<p class="status-line">暂无每日回顾</p>`;
  $("diary-page").textContent = `第 ${state.diary.page} / ${totalPages} 页`;
  $("diary-prev").disabled = state.diary.page <= 1;
  $("diary-next").disabled = state.diary.page >= totalPages;
  renderDiaryDetail(filtered.find((item) => item.cycle_date === state.diary.selectedDate));
}

function showActionStatus(message, kind = "") {
  const status = $("action-status");
  status.hidden = !message;
  status.textContent = message;
  status.classList.toggle("error", kind === "error");
}

function requestArchiveItem(kind, id, label, row) {
  state.pendingDelete = { kind, id, label, row };
  $("delete-confirm-message").textContent = `确定要删除这条${label}吗？删除后不会再注入，但历史审计仍会保留。`;
  $("delete-confirm").hidden = false;
  $("delete-apply").focus();
}

function closeDeleteConfirmation() {
  state.pendingDelete = null;
  $("delete-confirm").hidden = true;
}

function applyArchivedItem(pending, result) {
  const ledger = state.data?.ledger;
  if (!ledger) return;
  ledger.state_version = result.state_version ?? ledger.state_version;
  if (result.mood) ledger.mood = result.mood;
  if (pending.kind === "event") {
    const item = ledger.events.find((event) => event.id === pending.id);
    if (item) item.lifecycle = "archived";
  } else {
    const item = ledger.attention_items.find((attention) => attention.id === pending.id);
    if (item) item.status = "archived";
    const presented = state.data.presentation?.attention_items;
    if (Array.isArray(presented)) {
      state.data.presentation.attention_items = presented.filter(
        (attention) => attention.item_id !== pending.id,
      );
    }
  }

  if (pending.row && typeof pending.row.remove === "function") {
    pending.row.remove();
  }
  if (pending.kind === "event") {
    const events = ledger.events.filter((event) => event.lifecycle !== "archived");
    $("event-count").textContent = `${events.length} 条`;
    renderMood(ledger);
    renderEventOrbits(events);
  } else {
    const remaining = state.data.presentation?.attention_items || [];
    $("attention-count").textContent = `${remaining.length} 条`;
  }
}

async function archivePendingItem() {
  const pending = state.pendingDelete;
  if (!pending) return;
  closeDeleteConfirmation();
  showActionStatus(`正在删除${pending.label}…`);
  try {
    const result = await api("page/items/delete", {
      method: "POST",
      body: {
        session_id: state.sessionId,
        kind: pending.kind,
        id: pending.id,
      },
    });
    applyArchivedItem(pending, result);
    showActionStatus(`${pending.label}已删除`);
  } catch (error) {
    showActionStatus(`删除失败：${error.message}`, "error");
  }
}

function eventCategoryLabel(category) {
  return ({
    episodic: "近期片段",
    psychological: "心理余波",
    concrete: "长期心事",
  })[category] || "心事";
}

/* ===== 心境三档联动（暖/绷/沉）· 环仪表 · 轨道光点 · 昼夜 ===== */
const BAND_LINES = {
  warm:  "此刻她心里是软的，还留着一点没说完的话",
  tense: "她现在有点绷紧了，像一拧就着的弦",
  heavy: "她好像把自己折起来了，轻轻的，不敢展开",
};

function moodBand(mood) {
  const valence = Number(mood?.valence);
  const tension = Number(mood?.tension);
  if (!Number.isFinite(valence) || !Number.isFinite(tension)) return "warm";
  if (tension >= 0.5) return "tense";
  if (valence < 0.4) return "heavy";
  return "warm";
}

function categoryClass(category) {
  return ({
    episodic: "episodic",
    psychological: "psychological",
    concrete: "concrete",
  })[category] || "episodic";
}

function renderMood(ledger) {
  const band = moodBand(ledger.mood);
  document.documentElement.dataset.mood = band;
  $("mood-label").textContent = ledger.mood.label;
  $("hero-h2").textContent = BAND_LINES[band];
  const activeCount = (Array.isArray(ledger.events) ? ledger.events : [])
    .filter((item) => item.lifecycle !== "archived").length;
  const hint = $("core-hint");
  if (hint) {
    hint.textContent = activeCount > 0 ? `${activeCount} 条心事在打转` : "心里暂时很安静";
  }
  setDial("1", ledger.mood.valence);
  setDial("2", ledger.mood.energy);
  setDial("3", ledger.mood.tension);
  buildRain();
}

/* 环仪表：数值居中，进度弧 + 末端小亮珠位置由弧度精确计算 */
function setDial(index, value) {
  const track = $(`track-${index}`);
  const dot = $(`dot-${index}`);
  const val = $(`val-${index}`);
  if (!track) return;
  const v = Math.min(1, Math.max(0, Number(value) || 0));
  const circumference = 2 * Math.PI * 36; // r=36 → 226.19
  track.style.strokeDashoffset = String(circumference * (1 - v));
  const degree = v * 360 - 90;
  const radian = (degree * Math.PI) / 180;
  dot.setAttribute("cx", String(43 + 36 * Math.cos(radian)));
  dot.setAttribute("cy", String(43 + 36 * Math.sin(radian)));
  dot.setAttribute("r", v >= 0.985 ? "0" : "2.8");
  val.textContent = Number(value).toFixed(2);
}

function setBodyDial(intimacy) {
  const effective = Number(intimacy?.body_sensitivity) * 0.45
    + Number(intimacy?.sexual_arousal) * 0.55;
  setDial("4", Number.isFinite(effective) ? effective : 0);
}

function renderEventOrbits(events) {
  const palette = [
    "var(--accent)",
    "var(--mint)",
    "var(--gold)",
    "var(--lavender)",
    "var(--rose)",
    "var(--blue)",
  ];
  $("event-orbits").innerHTML = (events || []).slice(0, 8).map((item, index) => {
    const angle = (index / Math.max(events.length, 1)) * Math.PI * 2;
    const x = 50 + Math.cos(angle) * 38;
    const y = 50 + Math.sin(angle) * 38;
    const color = palette[index % palette.length];
    const fact = text(item.fact);
    const tip = fact.length > 26 ? `${fact.slice(0, 26)}…` : fact;
    return `<i class="sat" style="left:${x.toFixed(2)}%;top:${y.toFixed(2)}%;background:${color};color:${color}" data-tip="${html(tip)}"></i>`;
  }).join("");
}

function intimacyStageLabel(stage) {
  return ({
    not_noticeable: "身体平静，没有明显性反应",
    slightly_aware: "身体开始发热，敏感度和性唤起轻微上升",
    warmly_receptive: "身体反应明显，发热和敏感感持续增强",
    open_and_receptive: "身体反应强烈，敏感度和性唤起处于高位",
    warming: "身体开始发热，性唤起正在升起",
    close: "身体反应较明显，敏感感持续",
    aroused: "性唤起已经明显升高",
  })[stage] || "未知身体反应";
}

function bodyReactionStage(intimacy) {
  const effective = Number(intimacy.body_sensitivity) * 0.45
    + Number(intimacy.sexual_arousal) * 0.55;
  if (effective >= 0.58) return "open_and_receptive";
  if (effective >= 0.32) return "warmly_receptive";
  if (effective >= 0.08) return "slightly_aware";
  return "not_noticeable";
}

function renderGuidance(diagnostics) {
  const data = diagnostics?.expression_guidance || {};
  // 旧版后端的 payload 没有 can_say 键——区分"真的没内容"和"插件后端未重载"
  const backendStale = !("can_say" in data);
  const tone = text(data.tone).trim();
  const canSay = text(data.can_say).trim();
  const avoid = text(data.avoid).trim();
  const hasContent = Boolean(tone);
  const stateLabel = backendStale
    ? "后端待重载"
    : hasContent
      ? (data.will_inject ? "已激活" : "已生成 · 待注入")
      : (data.regime ? "缓存中" : "未生成");

  $("guidance-state").textContent = stateLabel;
  $("guidance-tone").textContent = tone || "此刻内心平静，按人格正常聊即可。";

  const rows = [
    ["可以流露", canSay, !canSay],
    ["避免", avoid, !avoid],
  ];
  $("guidance-rows").innerHTML = backendStale
    ? ""
    : rows.map(([label, value, empty]) => `
      <div class="guidance-row${empty ? " empty" : ""}">
        <span class="lb">${html(label)}</span>
        <span class="val">${empty ? "（无）" : html(value)}</span>
      </div>
    `).join("");

  const generatedAt = text(data.generated_at).trim();
  const trigger = text(data.trigger).trim();
  const source = backendStale
    ? "旧版后端"
    : data.model_generated ? "模型生成" : (hasContent ? "内嵌兜底" : "—");
  const bits = [];
  bits.push(source);
  if (trigger) bits.push(`触发：${trigger}`);
  if (generatedAt) bits.push(`更新于 ${formatBeijingTime(generatedAt)}`);
  $("guidance-note").innerHTML = backendStale
    ? "回复建议字段不完整 · 请在插件管理里重载「内心世界」后刷新页面"
    : hasContent
      ? `回复建议 · ${bits.join(" · ")}${data.will_inject ? "" : " · 当前不会注入"}`
      : "回复建议 · 内心平静，未生成具体表达建议";
}

function renderState(payload) {
  const ledger = payload.ledger;
  state.data = payload;
  $("empty").hidden = true;
  $("empty").classList.remove("error");
  $("workspace").hidden = false;
  renderMood(ledger);

  const events = ledger.events.filter((item) => item.lifecycle !== "archived");
  $("event-count").textContent = `${events.length} 条`;
  $("ev-total").textContent = String(events.length);
  $("events").innerHTML = events.length ? events.map((item) => `
    <article class="event-item">
      <i class="event-marker"></i><div><strong>${html(item.fact)}</strong><p class="mean">${html(item.emotional_meaning)}</p></div>
      <span class="event-state"><b class="etag ${html(categoryClass(item.category))}">${html(eventCategoryLabel(item.category))}</b><em>${html(item.lifecycle)}</em><button class="delete-item" type="button" data-delete-kind="event" data-delete-id="${html(item.id)}" aria-label="删除这条心事" title="删除这条心事">删除</button></span>
    </article>
  `).join("") : `<p class="status-line">暂无持续影响的事情</p>`;
  renderEventOrbits(events);

  const attentionItems = Array.isArray(payload.presentation?.attention_items)
    ? payload.presentation.attention_items
    : [];
  $("attention-count").textContent = `${attentionItems.length} 条`;
  $("attention-items").innerHTML = attentionItems.length ? attentionItems.map((item) => `
    <article class="event-item">
      <i class="event-marker"></i><div><strong>${html(item.content)}</strong><p class="mean">${html([
        item.time_hint || item.due_at,
        item.overdue ? "已到时间但尚无完成证据" : "",
      ].filter(Boolean).join(" · ") || "持续到明确完成、取消或替代")}</p></div>
      <span class="event-state"><b class="etag attention">${html(item.kind_label || item.kind)}</b><em>${html(item.status_label || item.status)}</em><button class="delete-item" type="button" data-delete-kind="attention" data-delete-id="${html(item.item_id)}" aria-label="删除这条待关注事项" title="删除这条待关注事项">删除</button></span>
    </article>
  `).join("") : `<p class="status-line">暂无待关注事项</p>`;

  const intimacy = ledger.intimacy;
  $("intimacy-tier").textContent = text(
    payload.presentation?.persona_intimacy_tier || "很亲密",
  );
  $("intimacy-stage").textContent = text(
    payload.presentation?.body_reaction_stage
      || intimacyStageLabel(bodyReactionStage(intimacy)),
  );
  $("intimacy").innerHTML = [
    ["身体敏感", intimacy.body_sensitivity],
    ["性唤起", intimacy.sexual_arousal],
  ].map(([label, value]) => `<div class="dimension"><span>${label}</span><i style="--value:${Math.round(Number(value) * 100)}%"></i><b>${Number(value).toFixed(2)}</b></div>`).join("");
  setBodyDial(intimacy);

  $("hero-desc").textContent =
    `有效心事 ${events.length} 条 · 待关注 ${attentionItems.length} 项 · 随聊天持续演化`;

  renderDiaries();

  const diagnostics = payload.diagnostics || {};
  renderGuidance(diagnostics);
  $("diagnostics").innerHTML = [
    ["Busy Schedule", diagnostics.busy_schedule ? "已连接" : "未连接"],
    ["LivingMemory", diagnostics.livingmemory ? "已连接" : "未连接"],
    ["主动对话 Spark", diagnostics.spark ? "已连接" : "未连接"],
    ["主动消息回应", diagnostics.spark_awaiting_user_reply
      ? `等待中 · ${Number(diagnostics.spark_waiting_minutes || 0).toFixed(1)} 分钟 · 已结算 ${Number(diagnostics.spark_applied_stage || 0)} 阶段`
      : "当前没有待回应消息"],
    ["低频复核链", (diagnostics.review_provider_chain || []).join(" → ")],
    ["每日深度回顾链", (diagnostics.daily_provider_chain || []).join(" → ")],
    ["最近实际提示词", diagnostics.last_actual_prompt_available ? "可查看" : "尚未产生"],
  ].map(([label, value]) => `<dt>${html(label)}</dt><dd>${html(value)}</dd>`).join("");
}

async function loadState() {
  state.sessionId = $("session-id").value;
  if (!state.sessionId) {
    showStatus("尚无可读取的私聊状态", "empty");
    return;
  }
  writeSessionId(state.sessionId);
  showStatus("正在载入私聊状态");
  try {
    renderState(await api("page/state", { params: { session_id: state.sessionId } }));
    await loadPrompt();
  } catch (error) {
    showStatus(`载入失败：${error.message}`, "error");
  }
}

function formatBeijingTime(value) {
  if (!value) return "时间未知";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  const parts = Object.fromEntries(new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  }).formatToParts(parsed).map(({ type, value: part }) => [type, part]));
  return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}:${parts.second}`
    + "（北京时间 UTC+08:00）";
}

function injectionMeta(snapshot, currentVersion) {
  if (!snapshot) return "尚未产生请求装配记录";
  const source = ({
    normal: "普通对话",
    spark_proactive: "Spark 主动对话",
    legacy: "旧缓存",
    preview: "当前账本实时预览",
  })[snapshot.request_source] || snapshot.request_source || "未知";
  const version = Number.isInteger(snapshot.state_version) && snapshot.state_version >= 0
    ? `v${snapshot.state_version}`
    : "版本未知";
  const stale = snapshot.stale || (
    Number.isInteger(currentVersion)
    && Number.isInteger(snapshot.state_version)
    && snapshot.state_version >= 0
    && snapshot.state_version !== currentVersion
  );
  const freshness = stale
    ? `历史请求时点，已落后于当前实时账本 v${currentVersion}`
    : "与当前实时账本版本一致";
  const timeLabel = snapshot.request_source === "preview" ? "预览生成" : "请求装配";
  return `${source} · ${version} · ${freshness} · ${timeLabel}于 ${formatBeijingTime(snapshot.generated_at)}`;
}

async function loadPrompt() {
  const payload = await api("page/injection", { params: { session_id: state.sessionId } });
  if (payload.actual !== undefined || payload.preview) {
    $("prompt-kind").textContent = injectionMeta(payload.actual, payload.current_state_version);
    $("prompt").textContent = payload.actual?.prompt || "尚未产生请求装配记录。";
    $("preview-kind").textContent = injectionMeta(payload.preview, payload.current_state_version);
    $("preview-prompt").textContent = payload.preview?.prompt || "当前预览不可用。";
    return;
  }

  $("prompt-kind").textContent = payload.kind === "actual"
    ? "旧版接口：最近一次请求装配"
    : "旧版接口：尚未产生请求装配";
  $("prompt").textContent = payload.kind === "actual" ? payload.prompt : "尚未产生请求装配记录。";
  $("preview-kind").textContent = payload.kind === "preview"
    ? "旧版接口：当前状态预览"
    : "旧版接口未提供当前状态预览";
  $("preview-prompt").textContent = payload.kind === "preview" ? payload.prompt : "";
}

async function runRule() {
  try {
    const payload = await api("page/rules/test", { method: "POST", body: { text: $("rule-sample").value } });
    $("rule-output").textContent = JSON.stringify(payload.run, null, 2);
  } catch (error) {
    $("rule-output").textContent = error.message;
  }
}

async function initialize() {
  showStatus("正在载入私聊状态");
  $("session-id").disabled = true;
  $("load").disabled = true;
  try {
    const payload = await api("page/sessions");
    if (!renderSessions(payload.sessions)) {
      showStatus("尚无私聊状态，收到一条普通私聊消息后会自动创建", "empty");
      return;
    }
    await loadState();
  } catch (error) {
    showStatus(`载入失败：${error.message}`, "error");
    $("load").disabled = false;
  }
}

function buildRain() {
  const rain = $("rain");
  if (!rain) return;
  rain.innerHTML = "";
  const heavy = document.documentElement.dataset.mood === "heavy";
  rain.style.display = heavy ? "block" : "none";
  if (!heavy) return;
  for (let index = 0; index < 36; index += 1) {
    const drop = document.createElement("i");
    drop.style.left = `${Math.random() * 100}%`;
    drop.style.animationDuration = `${2.5 + Math.random() * 3}s`;
    drop.style.animationDelay = `${Math.random() * 4}s`;
    drop.style.opacity = String(0.2 + Math.random() * 0.4);
    rain.appendChild(drop);
  }
}

function applyTheme(manual) {
  const root = document.documentElement;
  if (!manual) {
    const hour = new Date().getHours();
    root.dataset.theme = hour >= 7 && hour < 19 ? "day" : "night";
  }
  $("theme").textContent = root.dataset.theme === "night" ? "☀" : "☾";
}

$("theme").addEventListener("click", () => {
  const root = document.documentElement;
  root.dataset.theme = root.dataset.theme === "night" ? "day" : "night";
  applyTheme(true);
});
applyTheme(false);
buildRain();

$("load").addEventListener("click", initialize);
$("refresh-prompt").addEventListener("click", loadPrompt);
$("run-rule").addEventListener("click", runRule);
$("delete-cancel").addEventListener("click", closeDeleteConfirmation);
$("delete-apply").addEventListener("click", archivePendingItem);
$("delete-confirm").addEventListener("click", (event) => {
  if (event.target === $("delete-confirm")) closeDeleteConfirmation();
});
$("session-id").addEventListener("change", () => {
  state.diary.page = 1;
  state.diary.selectedDate = "";
  loadState();
});
$("diary-search").addEventListener("input", (event) => {
  state.diary.query = event.target.value.trim();
  state.diary.page = 1;
  state.diary.selectedDate = "";
  renderDiaries();
});
$("diary-month").addEventListener("change", (event) => {
  state.diary.month = event.target.value;
  state.diary.page = 1;
  state.diary.selectedDate = "";
  renderDiaries();
});
$("diary-prev").addEventListener("click", () => {
  state.diary.page -= 1;
  state.diary.selectedDate = "";
  renderDiaries();
});
$("diary-next").addEventListener("click", () => {
  state.diary.page += 1;
  state.diary.selectedDate = "";
  renderDiaries();
});
$("events").addEventListener("click", (event) => {
  const target = event.target.closest("[data-delete-kind]");
  if (!target) return;
  requestArchiveItem(
    target.dataset.deleteKind,
    target.dataset.deleteId,
    target.dataset.deleteKind === "event" ? "心事" : "待关注事项",
    target.closest(".event-item"),
  );
});
$("attention-items").addEventListener("click", (event) => {
  const target = event.target.closest("[data-delete-kind]");
  if (!target) return;
  requestArchiveItem(
    target.dataset.deleteKind,
    target.dataset.deleteId,
    "待关注事项",
    target.closest(".event-item"),
  );
});
$("diaries").addEventListener("click", (event) => {
  const target = event.target.closest("[data-diary-date]");
  if (!target) return;
  state.diary.selectedDate = target.dataset.diaryDate;
  renderDiaries();
});
initialize();