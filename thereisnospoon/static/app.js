"use strict";

const POLL_MS = 2000;
const $ = (id) => document.getElementById(id);

const SAMPLE_CHAT = [
  ["@neon_moth", "VJ, make the moon a disco ball and let it drip onto the city"],
  ["@quietkettle", "first time here, is this whole thing generated live??"],
  ["@pixelgoblin", "I want a cat piloting a submarine through a jelly ocean"],
  ["@delta_rune", "can it be raining tiny glowing umbrellas please"],
  ["@sk8rgrl", "lol the last one was so weird, more weird"],
  ["@orbitalfern", "make a forest where the trees hum in harmony"],
  ["@crt_ghost", "VHS vibes, 1987 late night cable access show energy"],
  ["@marmalade", "somebody give the narrator a tiny hat"],
  ["@vortexvera", "slow dramatic dolly zoom on a confused penguin"],
  ["@byte_bard", "end it with everyone dancing on a giant spoon"],
];

const METRICS = [
  ["grid_16x16_ratio", "16x16 grid ratio", "lower", "< 1.15", (v) => v < 1.15],
  ["temporal_motion_mean_l1", "Motion fluidity (mean |Δ|)", "higher", "", null],
  ["active_motion_frame_fraction", "Active motion frames", "higher", "", null],
  ["temporal_accel_ratio", "Temporal accel ratio", "lower", "", null],
  ["vae_17f_pulse_ratio", "VAE 17-frame pulse", "lower", "≈ 1.0", null],
  ["mean_laplacian_sharpness", "Laplacian sharpness", "higher", "", null],
  ["audio_rms_dbfs", "Audio RMS dBFS", "", "-30 … -12", (v) => v >= -30 && v <= -12],
];

const TELEMETRY = [
  ["is_streaming", "streaming"], ["clips_broadcasted", "clips sent"], ["bytes_sent", "bytes sent"],
  ["uptime_sec", "uptime"], ["queue_size", "queued"], ["rtmp_url_masked", "target"],
  ["current_clip_path", "current clip"], ["last_error", "last error"],
];

let lastClipUrl = null;

async function api(path, body) {
  const json = {"Content-Type": "application/json"};
  const opts = body === undefined ? {} : {method: "POST", headers: json, body: JSON.stringify(body)};
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ? JSON.stringify(data.detail) : `${res.status} ${res.statusText}`);
  return data;
}

function setStatus(id, text, kind) {
  const el = $(id);
  el.textContent = text || "";
  el.className = "status-line" + (kind ? " " + kind : "");
}

async function action(btn, statusId, label, fn) {
  btn.disabled = true;
  setStatus(statusId, label + "…");
  try {
    const out = await fn();
    setStatus(statusId, out || label + " ✓", "ok");
    await refresh();
  } catch (err) {
    setStatus(statusId, label + " failed: " + err.message, "err");
  } finally {
    btn.disabled = false;
  }
}

const isBlank = (v) => v === null || v === undefined || v === "";

function fmtNum(v, digits) {
  if (isBlank(v)) return "—";
  if (typeof v !== "number") return String(v);
  return Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(digits === undefined ? 3 : digits);
}

function fmtBytes(n) {
  if (typeof n !== "number") return fmtNum(n);
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return n.toFixed(i ? 1 : 0) + " " + units[i];
}

function fmtDuration(s) {
  if (typeof s !== "number") return fmtNum(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return (h ? h + "h " : "") + (h || m ? m + "m " : "") + sec + "s";
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setPill(id, text, kind) {
  $(id).textContent = text;
  $(id).className = "pill " + kind;
}

function renderChannel(ch, chat, rtmp, srv) {
  $("channel-link").href = ch.url;
  $("channel-link").textContent = ch.handle;
  $("channel-id").textContent = ch.id;
  const strange = chat.mode === "creative_and_strange_no_chat";
  setPill("pill-mode", "mode: " + chat.mode, strange ? "strange" : "ok");
  const live = Boolean(rtmp && rtmp.is_streaming);
  setPill("pill-live", live ? "● LIVE on YouTube" : "RTMP: idle", live ? "live" : "");
  setPill("pill-server", "server: online", "ok");
  $("uptime").textContent = "server uptime " + fmtDuration(srv.uptime_sec) + " · v" + srv.version;
}

function renderChat(chat) {
  const list = $("chat-list");
  list.replaceChildren();
  $("chat-count").textContent = chat.count + " / 10";
  if (!chat.lines.length) {
    list.appendChild(el("li", "empty", "No chat yet — the VJ is in Creative & Strange mode."));
    return;
  }
  for (const line of chat.lines) {
    const li = el("li");
    li.append(el("span", "who", line.author || "anon"), el("span", "msg", line.message || ""),
      el("span", "src", line.source || ""));
    list.appendChild(li);
  }
  list.scrollTop = list.scrollHeight;
}

function renderVJ(vj, chat) {
  const strange = (vj.mode || chat.mode) === "creative_and_strange_no_chat";
  $("persona").textContent = vj.persona || chat.persona;
  $("persona").className = "persona" + (strange ? " strange" : "");
  $("vj-mode").textContent = vj.mode || chat.mode;
  $("vj-scene").textContent = vj.scene || "— synthesize a prompt to see the VJ's scene —";
  $("vj-dialogue").textContent = vj.dialogue_tag || "—";
  $("vj-system").textContent = vj.system_prompt || "";
  $("vj-prompt").textContent = vj.prompt || "";
  const bits = [];
  if (vj.generated_at) bits.push("generated " + new Date(vj.generated_at * 1000).toLocaleTimeString());
  if (typeof vj.elapsed_sec === "number") bits.push(vj.elapsed_sec.toFixed(1) + " s");
  if (vj.source) bits.push(vj.source);
  if (vj.busy) bits.push("synthesizing…");
  $("vj-meta").textContent = bits.join(" · ");
  if (vj.error) setStatus("vj-status", vj.error, "err");
}

function renderClip(clip) {
  const video = $("player");
  $("player-empty").classList.toggle("hidden", Boolean(clip.url));
  if (clip.url && clip.url !== lastClipUrl) {
    lastClipUrl = clip.url;
    video.src = clip.url;
    video.load();
    video.play().catch(() => {});
  }
  $("clip-name").textContent = clip.filename || "—";
  $("clip-size").textContent = clip.size_bytes ? fmtBytes(clip.size_bytes) : "";
  for (const [id, key] of [["clip-sheet", "sheet_url"], ["clip-receipt", "receipt_url"]]) {
    $(id).classList.toggle("hidden", !clip[key]);
    if (clip[key]) $(id).href = clip[key];
  }
  $("btn-render").disabled = Boolean(clip.busy);
  if (clip.busy) setStatus("clip-status", "Rendering 10 s clip on GPU… " + (clip.stage || ""));
  else if (clip.error) setStatus("clip-status", clip.error, "err");
}

function cellClass(dir, clean, ref, pass) {
  if (typeof clean !== "number") return "neutral";
  if (pass) return pass(clean) ? "good" : "badv";
  if (typeof ref !== "number" || !dir) return "neutral";
  return (dir === "lower" ? clean <= ref : clean >= ref) ? "good" : "badv";
}

function renderQuality(q) {
  const body = $("hud-body");
  body.replaceChildren();
  const clean = q.clean || {}, ref = q.study_v2 || {};
  for (const [key, label, dir, target, pass] of METRICS) {
    const c = clean[key], r = ref[key];
    const delta = typeof c === "number" && typeof r === "number" ? c - r : null;
    const tr = el("tr");
    tr.append(el("td", "", label), el("td", cellClass(dir, c, r, pass), fmtNum(c)), el("td", "neutral", fmtNum(r)),
      el("td", "neutral", fmtNum(delta)), el("td", "neutral", target || (dir ? dir + " is better" : "")));
    body.appendChild(tr);
  }
  const probe = q.ffprobe_ok === undefined ? "" : (q.ffprobe_ok ? "ffprobe spec ✓ · " : "ffprobe spec ✗ · ");
  const refs = "Study v2 refs: " + (q.study_v2_clips || 0);
  if (q.busy) setStatus("quality-status", "Measuring clip quality…");
  else if (q.error) setStatus("quality-status", q.error, "err");
  else setStatus("quality-status", probe + refs + (q.clip ? " · " + q.clip : ""), q.ffprobe_ok ? "ok" : "");
}

function renderRtmp(rtmp) {
  const dl = $("rtmp-telemetry");
  dl.replaceChildren();
  for (const [key, label] of TELEMETRY) {
    let v = rtmp[key];
    if (key === "bytes_sent") v = fmtBytes(v);
    else if (key === "uptime_sec") v = fmtDuration(v);
    else if (key === "current_clip_path" && v) v = String(v).split("/").pop();
    else if (typeof v === "boolean") v = v ? "yes" : "no";
    const box = el("div");
    const dd = el("dd", "", isBlank(v) ? "—" : String(v));
    dd.title = dd.textContent;
    box.append(el("dt", "", label), dd);
    dl.appendChild(box);
  }
  if (rtmp.rtmp_url_masked) $("rtmp-target").textContent = rtmp.rtmp_url_masked;
}

function renderComponents(comps) {
  const box = $("components");
  box.replaceChildren();
  for (const [name, state] of Object.entries(comps)) {
    const ok = state === "ok";
    const pill = el("span", "pill " + (ok ? "ok" : state === "not loaded" ? "" : "bad"), name + ": " + state);
    pill.title = state;
    box.appendChild(pill);
  }
}

async function refresh() {
  try {
    const s = await api("/api/status");
    renderChannel(s.channel, s.chat, s.rtmp, s.server);
    renderChat(s.chat);
    renderVJ(s.vj, s.chat);
    renderClip(s.clip);
    renderQuality(s.quality);
    renderRtmp(s.rtmp);
    renderComponents(s.components);
  } catch (err) {
    setPill("pill-server", "server: offline", "bad");
  }
}

function streamBody(act) {
  return {action: act, rtmp_url: $("rtmp-url").value.trim(), stream_key: $("rtmp-key").value.trim()};
}

function bind() {
  const inject = (body, msg) => api("/api/chat/inject", body).then(() => msg);
  const handlers = {
    "btn-sample": ["chat-status", "Load samples", () => inject({lines: SAMPLE_CHAT.map(([author, message]) =>
      ({author, message}))}, "Loaded 10 sample lines → Chat VJ mode ✓")],
    "btn-clear": ["chat-status", "Clear", () => inject({clear: true}, "Chat cleared → Creative & Strange mode ✓")],
    "btn-poll": ["chat-status", "Live poll", () => api("/api/chat/poll", {}).then((r) =>
      `Polled @THEREISNOSPOON-t4j: ${r.new_messages} new · live=${r.live}` + (r.note ? " · " + r.note : ""))],
    "btn-preview": ["vj-status", "Synthesize", () => api("/api/vj/preview-prompt", {}).then((r) =>
      `Fresh ${r.persona} prompt in ${r.elapsed_sec.toFixed(1)} s ✓`)],
    "btn-render": ["clip-status", "Render", () => api("/api/clip/generate", {}).then((r) => r.message)],
    "btn-quality": ["quality-status", "Re-measure", () => api("/api/quality/refresh", {}).then((r) => r.message)],
  };
  const streamButtons = [["start", "btn-stream-start"], ["stop", "btn-stream-stop"], ["configure", "btn-stream-save"]];
  for (const [act, id] of streamButtons) {
    handlers[id] = ["rtmp-status", act, () => api("/api/stream/configure", streamBody(act)).then((r) =>
      `${act}: streaming=${r.telemetry.is_streaming} → ${r.telemetry.rtmp_url_masked || "unconfigured"}`)];
  }
  for (const [id, [statusId, label, fn]] of Object.entries(handlers)) {
    $(id).onclick = (ev) => action(ev.currentTarget, statusId, label, fn);
  }
  $("chat-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const message = $("chat-message").value.trim();
    const author = $("chat-author").value.trim() || "@console_viewer";
    if (message) {
      action(ev.submitter || $("chat-form"), "chat-status", "Send", () => inject({lines: [{author, message}]},
        "Sent to the last-10 window ✓").then((out) => { $("chat-message").value = ""; return out; }));
    }
  });
  $("rtmp-key").addEventListener("input", () => {
    const base = $("rtmp-url").value.trim().replace(/\/+$/, "");
    const key = $("rtmp-key").value.trim();
    $("rtmp-target").textContent = base + "/" + (key ? "••••" + key.slice(-4) : "…");
  });
}

bind();
refresh();
setInterval(refresh, POLL_MS);
