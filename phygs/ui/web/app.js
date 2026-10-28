// Spot Robot UI. Protocol: phygs/ui/server.py; design: docs/design/ui/design.html.
// The task and Start; move the base (Navigate to, Drive, Stop); move the arm (joints, path, Stow,
// gripper, Look at, Move to pose, Move along, plan preview); cancel; End; E-stop.
// With a replay server (phygs/ui/replay_cli.py) the same page plays a recording, read-only.
"use strict";

const VIEW_LABELS = { front: "Front", map: "Map", hand: "Hand", frontleft: "Front-left", frontright: "Front-right" };
const CHANNEL_OF_KIND = { manipulation: "arm", navigation: "base", locomotion: "base" };
const SYMBOL = { executing: "●", succeeded: "✓", failed: "✗", cancelled: "⊘", armed: "◆" };
const SKILL_GROUPS = [
  ["MOVE THE BASE", ["Navigate to", "Drive", "Stop"]],
  ["ARM AND GRIPPER", ["Arm joints", "Joint path", "Stow", "Open", "Close", "Look at", "Move to pose", "Move along"]],
];
const JOINT_LABELS = {
  arm_sh0: "Shoulder yaw", arm_sh1: "Shoulder pitch", arm_el0: "Elbow pitch",
  arm_el1: "Elbow roll", arm_wr0: "Wrist pitch", arm_wr1: "Wrist roll",
};
const ARM = Object.keys(JOINT_LABELS);
const NUDGE_RAD = 0.05;
const DRIVE_KEYS = { w: [1, 0, 0], s: [-1, 0, 0], a: [0, 1, 0], d: [0, -1, 0], q: [0, 0, 1], e: [0, 0, -1] };
const MIN_HEADING_DRAG_M = 0.15; // A shorter drag on the map sets no heading.
const DEG = 180 / Math.PI;
const ICONS = { // 16 px line icons, stroked with the text colour.
  navigate_to: '<circle cx="8" cy="8" r="6"/><circle cx="8" cy="8" r="2"/><path d="M8 0v3M8 13v3M0 8h3M13 8h3"/>',
  drive: '<path d="M8 1l3 3H5zM8 15l3-3H5zM1 8l3-3v6zM15 8l-3-3v6z"/>',
  stop: '<path d="M5 1h6l4 4v6l-4 4H5l-4-4V5z"/><path d="M5 8h6"/>',
  move_joints: '<path d="M2 4h12M2 8h12M2 12h12"/><circle cx="5" cy="4" r="1.6"/><circle cx="11" cy="8" r="1.6"/><circle cx="7" cy="12" r="1.6"/>',
  execute_trajectory: '<path d="M2 13l4-6 4 3 4-7"/><circle cx="2" cy="13" r="1.4"/><circle cx="6" cy="7" r="1.4"/><circle cx="10" cy="10" r="1.4"/><circle cx="14" cy="3" r="1.4"/>',
  stow: '<path d="M3 3l5 5 5-5M3 9l5 5 5-5"/>',
  unstow: '<path d="M3 13l5-5 5 5M3 7l5-5 5 5"/>',
  open_gripper: '<path d="M3 2v5l3 3v4M13 2v5l-3 3v4"/>',
  close_gripper: '<path d="M5 2v5l2 3v4M11 2v5l-2 3v4"/>',
  look_at: '<path d="M1 8s2.5-5 7-5 7 5 7 5-2.5 5-7 5-7-5-7-5z"/><circle cx="8" cy="8" r="2.2"/>',
  move_to_pose: '<circle cx="11" cy="5" r="3"/><path d="M2 14l6-6M2 14h4M2 14v-4"/>',
  move_along: '<path d="M8 15V2M4 6l4-4 4 4"/>',
};
const OUTCOMES = { // How the episode ended, as the page names it; the judge's verdict is shown with it.
  ended: ["Ended by you", "cancelled"], budget: ["Time budget spent", "cancelled"],
  interrupted: ["E-stop (interrupted)", "failed"], failed: ["Failed", "failed"], completed: ["Completed", "cancelled"],
};
const TARGET_SKILLS = ["look_at", "move_to_pose"];

const ui = {
  socket: null, hello: null, state: null,
  skills: new Map(),
  view: { main: "front", mode: "rgb" },
  images: {}, map: null,
  mapFit: null, imageFit: null,
  mapView: { zoom: 1, center: null, follow: false },
  pan: null,
  cursor: null, probe: null, probeSent: 0,
  trail: [],
  commands: new Map(),
  logFilter: "all",
  armed: null,
  goal: null, drag: null, lastGoal: null,
  drive: { keys: new Set(), pad: [0, 0, 0], fraction: 0.5, blocked: false },
  target: null, // clicked pixel { view, tick, x, y } for a target skill
  targetFound: null, // what the server resolved there
  choices: {}, // per skill input: the chosen value
  joints: null, // Arm joint targets being edited [rad]
  jointDuration: "",
  poses: [], // saved arm poses: { name, positions, seconds }
  timelineHits: [],
  replay: null, // { duration, speeds } when a replay server plays a recording
  resultClosed: false,
  endArmed: 0, // End episode: the time of the first click; a second within 3 s confirms
  preview: null, // the plan shown for the selected skill: { id, skill, status, duration, path_body, arm_body, path_world, reason }
  telemetry: { tab: "health", history: [], latest: null, drawn: 0, inspect: null, bytes: 0, frames: 0, rate: null },
  link: { opened: false, failures: 0 },
};

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children.filter((c) => c != null));
  return node;
};
const fmt = (value, digits = 2) => (value == null || Number.isNaN(value) ? "–" : Number(value).toFixed(digits));
const icon = (name) => {
  const span = el("span", { className: "icon" });
  span.innerHTML = `<svg viewBox="0 0 16 16" aria-hidden="true">${ICONS[name] || ""}</svg>`;
  return span;
};

// ---------- connection ----------
function connect() {
  const token = new URLSearchParams(location.search).get("token") || "";
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/ws?token=${encodeURIComponent(token)}`);
  socket.binaryType = "arraybuffer";
  ui.socket = socket;
  socket.onopen = () => {
    ui.link = { opened: true, failures: 0 };
    $("connection-card").hidden = true;
    setConnection(true, "connected");
  };
  socket.onclose = (event) => {
    setConnection(false, event.reason || "offline, reconnecting");
    ui.link.failures += 1;
    showConnectionProblem(event.code);
    if (event.code !== 4001 && event.code !== 4003) setTimeout(connect, 2000);
  };
  socket.onmessage = (event) => {
    const binary = typeof event.data !== "string";
    ui.telemetry.bytes += binary ? event.data.byteLength : event.data.length;
    if (binary) ui.telemetry.frames += 1;
    if (!binary) onText(JSON.parse(event.data));
    else onBinary(event.data);
  };
}

// Why the page shows nothing: a wrong or missing token, a second browser, or no server to reach.
function showConnectionProblem(code) {
  const port = location.port || (location.protocol === "https:" ? "443" : "80");
  let title = null, text = "", command = null;
  if (code === 4003) {
    title = "This link has no valid access token";
    text = "Open the full link the launcher printed, including ?token=… The token is new for every run.";
  } else if (code === 4001) {
    title = "Another browser is controlling this run";
    text = "Only one page may connect at a time. Close the other tab or window, then reload this one.";
  } else if (!ui.link.opened && ui.link.failures >= 2) {
    title = "Cannot reach the robot UI";
    text = "Check that cli.py is still running. If this browser runs on another computer than the simulation, forward the port first (VS Code: Ports > Forward a Port), then open the same link:";
    command = `ssh -L ${port}:127.0.0.1:${port} <user>@<simulation machine>`;
  }
  if (!title) return;
  $("connection-title").textContent = title;
  $("connection-text").textContent = text;
  $("connection-command").hidden = !command;
  $("connection-command").textContent = command || "";
  $("connection-card").hidden = false;
}

function send(message) {
  if (ui.replay && message.type !== "replay") return; // A replay is read-only.
  if (ui.socket && ui.socket.readyState === WebSocket.OPEN) ui.socket.send(JSON.stringify(message));
}

function setConnection(on, text) {
  const node = $("connection");
  node.textContent = text;
  node.className = `conn ${on ? "on" : "off"}`;
}

// ---------- messages ----------
function onText(message) {
  switch (message.type) {
    case "hello": onHello(message); break;
    case "state": onState(message); break;
    case "step": onStep(message); break;
    case "command": onCommand(message); break;
    case "episode": onEpisode(message); break;
    case "probe":
      ui.probe = message;
      if (ui.target && message.view === ui.target.view && message.tick === ui.target.tick && message.x === ui.target.x && message.y === ui.target.y) {
        ui.targetFound = message.found;
        renderSkillPanel();
      }
      break;
    case "error": log(simNow(), `✗ ${message.message}`, "fail", "all"); break;
    case "preview": onPreview(message); break;
    case "replay_view": ui.view.main = message.main; ui.view.mode = message.mode || "rgb"; markTabs(); break;
    case "replay_clock": onReplayClock(message); break;
    case "replay_restart": location.reload(); break;
    default: break;
  }
}

function onBinary(buffer) {
  const view = new DataView(buffer);
  const length = view.getUint32(0, false);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, length)));
  const type = header.type === "map" ? "image/png" : "image/jpeg";
  createImageBitmap(new Blob([new Uint8Array(buffer, 4 + length)], { type })).then((bitmap) => {
    if (header.type === "map") {
      if (ui.map) ui.map.bitmap.close();
      ui.map = { bitmap, header };
    } else {
      const old = ui.images[header.view];
      if (old) old.bitmap.close();
      ui.images[header.view] = { bitmap, header, receivedAt: performance.now() };
    }
  });
}

function onHello(hello) {
  ui.hello = hello;
  ui.replay = hello.replay || null;
  document.body.classList.toggle("replay", Boolean(ui.replay));
  if (ui.replay) buildReplayBar();
  ui.skills = new Map((hello.skills || []).map((skill) => [skill.name, skill]));
  $("episode").textContent = `Episode ${hello.episode_id}`;
  $("task").textContent = hello.task || "–";
  $("task").title = hello.task || "";
  buildStartCard();
  buildTabs([...hello.views.slice(0, 1), "map", ...hello.views.slice(1)]);
  buildSkills();
  renderSkillPanel();
  sendView();
}

function onState(state) {
  ui.state = state;
  const phase = state.phase || "running";
  document.body.dataset.phase = phase;
  const pill = $("status");
  pill.textContent = phase === "ready" ? "ready" : state.status;
  pill.className = `pill ${phase === "ready" ? "s-armed" : state.status === "running" ? "s-executing" : state.status === "completed" ? "s-succeeded" : "s-failed"}`;
  $("start-card").hidden = phase !== "ready" || Boolean(ui.replay);
  $("end-episode").hidden = phase !== "running" || Boolean(ui.replay);
  $("estop").hidden = phase === "ended";
  renderTaskChip(state);
  followLookAround(state);
  recordTelemetry(state);
  $("clock-sim").textContent = `${fmt(state.sim_time, 1)} s`;
  $("clock-wall").textContent = formatWall(state.wall_time);
  $("clock-rtf").textContent = state.rtf == null ? "–" : `×${fmt(state.rtf, 2)}`;
  const pose = state.pose;
  if (pose) {
    $("r-pos").textContent = `${fmt(pose.x)}, ${fmt(pose.y)} m`;
    $("r-yaw").textContent = `${fmt(pose.yaw * DEG, 0)}°`;
    $("needle").setAttribute("transform", `rotate(${90 - pose.yaw * DEG})`);
    $("r-speed").textContent = `${fmt(state.speed)} m/s`;
    const last = ui.trail[ui.trail.length - 1];
    if (!last || Math.hypot(pose.x - last[0], pose.y - last[1]) > 0.05) {
      ui.trail.push([pose.x, pose.y]);
      if (ui.trail.length > 4000) ui.trail.shift();
    }
  }
  $("robot-loc").textContent = pose ? "true pose" : "no pose";
  if (state.arm) {
    $("r-arm").textContent = state.arm.stowed == null ? "–" : state.arm.stowed ? "stowed" : "not stowed";
    const gripper = state.arm.gripper;
    $("r-grip").textContent = gripper == null ? "–" : `${fmt(gripper)} rad`;
    if (gripper != null) $("r-grip-bar").style.width = `${Math.min(100, Math.max(0, (-gripper / 1.4) * 100))}%`;
    if (ui.armed === "move_joints") refreshJointNow();
  }
  $("r-hold").textContent = state.holding == null ? "–" : state.holding ? "yes" : "no";
  refreshCards();
}

function onStep(step) {
  const channel = CHANNEL_OF_KIND[step.kind] || "base";
  const active = activeCommand(channel);
  if (active) active.step = step.text;
  log(step.sim_time, step.text, "", channel);
  refreshCards();
}

function onCommand(event) {
  const isNew = !ui.commands.has(event.id);
  const record = ui.commands.get(event.id) || { ...event, step: "" };
  Object.assign(record, { status: event.status, state: event.state, feedback: event.feedback });
  if (event.state !== "executing" && record.ended == null) record.ended = event.sim_time;
  ui.commands.set(event.id, record);
  if (event.skill === "navigate_to" && event.inputs && event.inputs.x != null) {
    ui.lastGoal = { x: event.inputs.x, y: event.inputs.y, heading: event.inputs.heading, id: event.id };
    const pose = ui.state && ui.state.pose;
    if (isNew && pose) record.startDistance = Math.hypot(event.inputs.x - pose.x, event.inputs.y - pose.y);
  }
  if (event.skill === "drive" && event.state === "failed") ui.drive.blocked = true; // Until the keys are released.
  refreshCards();
  refreshTiles();
  if (ui.armed === event.skill) renderSkillPanel();
  if (event.state === "executing" && record.loggedStart) return;
  record.loggedStart = true;
  const cls = { succeeded: "ok", failed: "fail", cancelled: "cancel" }[event.state] || "";
  const text = `${SYMBOL[event.state] || ""} ${event.skill} ${resultText(record)}`;
  log(event.sim_time, text, cls, event.channel, event.id);
  const fresh = !ui.state || event.sim_time >= ui.state.sim_time - 2; // Not replayed history after a reconnect.
  if (fresh && event.state !== "executing" && event.skill !== "drive") toast(text, event.state);
}

// A planned preview: planning, then ready (drawn in the Front view and on the map) or the raw failure.
function onPreview(event) {
  if (!ui.preview || ui.preview.skill !== event.skill || (ui.preview.id && ui.preview.id !== event.id)) return;
  ui.preview = { ...ui.preview, ...event };
  if (event.status === "failed") log(event.sim_time, `✗ ${event.skill} plan: ${event.reason}`, "fail", "arm");
  renderSkillPanel();
}

function requestPreview(skill, inputs) {
  ui.preview = { skill: skill.name, status: "sent" };
  send({ type: "preview", skill: skill.name, inputs });
  renderSkillPanel();
}

function runPreview() {
  const preview = ui.preview;
  if (!preview || preview.status !== "ready") return;
  send({ type: "run", skill: preview.skill, inputs: { preview: preview.id } });
  disarm();
}

// The plan block under a previewable skill: Preview, its status, Run plan.
function previewBlock(skill, inputs, ready) {
  if (!skill.preview) return null;
  const preview = ui.preview && ui.preview.skill === skill.name ? ui.preview : null;
  let text = "Plan first to see the gripper path in the Front view and on the map, then run that exact plan.";
  let cls = "";
  if (preview && (preview.status === "sent" || preview.status === "planning")) { text = "Planning around what the cameras see…"; cls = "s-executing"; }
  if (preview && preview.status === "ready") { text = `Plan ready: ${fmt(preview.duration, 1)} s; the gripper path is drawn in the Front view and on the map.`; cls = "s-succeeded"; }
  if (preview && preview.status === "failed") { text = `✗ ${preview.reason}`; cls = "s-failed"; }
  // From the stow a plan first unfolds the arm straight to the pre-stow posture (the planner's model
  // calls the folded stow a self-collision).
  const stowed = Boolean(ui.state && ui.state.arm && ui.state.arm.stowed);
  if (stowed && !preview) text += " From the stow the arm first unfolds to the ready posture.";
  const box = el("div", { className: `plan-box ${cls}` }, el("div", { className: "small", textContent: text }));
  box.append(actions([
    ["Preview plan", () => requestPreview(skill, inputs()), "", !ready],
    ["Run this plan", runPreview, "primary", !(preview && preview.status === "ready")],
  ]));
  return box;
}

function onEpisode(event) {
  if (event.phase === "running") {
    log(event.sim_time, "▶ episode started", "ok", "all");
    return;
  }
  const [title, state] = OUTCOMES[event.outcome] || [event.outcome || event.status, "failed"];
  log(event.sim_time, `■ episode ended: ${event.outcome} · ${event.reason}`, state === "succeeded" ? "ok" : "fail", "all");
  if (ui.armed) disarm();
  showResult(event, title, state);
}

// ---------- episode ----------
function buildStartCard() {
  const hello = ui.hello, episode = hello.episode || {};
  $("start-episode").textContent = hello.episode_id;
  $("start-task").textContent = hello.task || "–";
  const facts = $("start-facts");
  facts.textContent = "";
  const fact = (name, value) => facts.append(el("div", {}, el("dt", { textContent: name }), el("dd", { textContent: value })));
  fact("Episode", episode.spec ? `${episode.spec} · ${hello.scene}` : hello.scene || "–");
  if (episode.goal && episode.goal !== "none") fact("Goal", `${episode.goal}: ${episode.description}`);
  if (episode.budget != null) fact("Time budget", `${fmt(episode.budget, 0)} s of simulated time`);
  fact("Tier", episode.tier || (episode.actions || []).join(", ") || "–");
  fact("Privileged help", episode.grants_used && episode.grants_used.length ? episode.grants_used.join(", ") : "none");
  fact("Skills", `${(hello.skills || []).length} · arm planner: ${hello.arm_backend || "–"}`);
}

function startEpisode() {
  send({ type: "episode", action: "start" });
}

function endEpisode() {
  const button = $("end-episode");
  if (performance.now() - ui.endArmed < 3000) {
    send({ type: "episode", action: "end" });
    ui.endArmed = 0;
    button.textContent = "End episode";
    return;
  }
  ui.endArmed = performance.now();
  button.textContent = "Click again to end";
  setTimeout(() => { if (performance.now() - ui.endArmed >= 3000) button.textContent = "End episode"; }, 3100);
}

function showResult(event, title, state) {
  const report = event.report || {};
  $("result-episode").textContent = report.episode_id != null ? report.episode_id : "–";
  const outcome = $("result-outcome");
  outcome.textContent = title;
  outcome.className = `outcome s-${state}`;
  $("result-reason").textContent = event.reason || "";
  const facts = $("result-facts");
  facts.textContent = "";
  const fact = (name, value) => facts.append(el("div", {}, el("dt", { textContent: name }), el("dd", { textContent: value })));
  fact("Simulated time", `${fmt(report.sim_seconds, 1)} s`);
  fact("Wall time", report.wall_seconds == null ? "–" : formatWall(report.wall_seconds));
  const counts = Object.values(report.commands || {});
  const total = counts.reduce((sum, byState) => sum + Object.values(byState).reduce((a, b) => a + b, 0), 0);
  const failed = counts.reduce((sum, byState) => sum + (byState.failed || 0), 0);
  fact("Commands", `${total} · ${failed} failed or refused`);
  fact("Drive time", `${fmt(report.drive_seconds, 1)} s`);
  if (report.judge) fact("Judge", report.judge.verdict);
  $("result-card").hidden = ui.resultClosed;
}

// The task and its time left, in simulated and (estimated) wall time.
function renderTaskChip(state) {
  const chip = $("task-chip"), task = state.task;
  chip.hidden = !task || task.budget == null;
  if (chip.hidden) return;
  const left = Math.max(0, task.budget - task.elapsed);
  const wall = state.rtf && state.phase === "running" ? ` · ≈${formatWall(left / state.rtf)} wall` : "";
  const what = task.goal === "retrieve" ? `Retrieve: ${task.description}` : "Explore";
  chip.textContent = `${what} · ${fmt(left, 0)} s left${wall}`;
  chip.classList.toggle("low", left < 0.1 * task.budget);
}

// ---------- replay ----------
function buildReplayBar() {
  $("replay-bar").hidden = false;
  const group = $("replay-speed");
  group.textContent = "";
  for (const speed of ui.replay.speeds) {
    const button = el("button", { type: "button", textContent: `×${speed}` });
    button.dataset.speed = String(speed);
    button.addEventListener("click", () => send({ type: "replay", action: "speed", speed }));
    group.append(button);
  }
}

function onReplayClock(clock) {
  ui.replayClock = clock;
  $("replay-clock").textContent = `${formatWall(clock.position)} / ${formatWall(clock.duration)}`;
  $("replay-play").textContent = clock.paused ? "Play" : "Pause";
  for (const button of document.querySelectorAll("#replay-speed button")) button.classList.toggle("on", Number(button.dataset.speed) === clock.speed);
}

// The raw result: status, failure code and reason, as the API returned them.
function resultText(record) {
  const feedback = record.feedback || {};
  let text = record.status;
  if (feedback.failure) text += ` ${feedback.failure}`;
  if (feedback.reason) text += `: ${feedback.reason}`;
  return text;
}

function inputsText(record) {
  const i = record.inputs || {};
  if (record.skill === "navigate_to" && i.x != null) {
    return `to (${fmt(i.x)}, ${fmt(i.y)}) m, ${i.heading == null ? "any heading" : `${fmt(i.heading * DEG, 0)}°`}`;
  }
  if (record.skill === "drive" && i.v_x != null) return `${fmt(i.v_x)}, ${fmt(i.v_y)} m/s · ${fmt(i.v_rot)} rad/s`;
  if (record.skill === "move_joints" && i.duration != null) return `in ${fmt(i.duration, 1)} s`;
  if (record.skill === "execute_trajectory" && i.waypoints) return `${i.waypoints.length} poses`;
  if (i.target && i.target.point) return `at (${i.target.point.map((v) => fmt(v)).join(", ")}) m`;
  if (record.skill === "move_along" && i.distance_m != null) return `${fmt(i.distance_m)} m`;
  if (i.preview) return `previewed plan ${i.preview}`;
  return "";
}

// ---------- views ----------
function buildTabs(views) {
  const tabs = $("tabs");
  tabs.textContent = "";
  for (const name of views) {
    const canvas = el("canvas", { width: 208, height: 116 });
    const button = el("button", { type: "button", className: "tab" }, canvas, el("span", { textContent: VIEW_LABELS[name] || name }));
    button.dataset.view = name;
    button.setAttribute("role", "tab");
    button.addEventListener("click", () => selectView(name));
    tabs.append(button);
  }
  markTabs();
}

function selectView(name) {
  ui.view.main = name;
  ui.cursor = null;
  ui.probe = null;
  markTabs();
  sendView();
}

function markTabs() {
  const skill = ui.armed && ui.skills.get(ui.armed);
  const picker = skill ? skill.picker : null;
  for (const tab of document.querySelectorAll(".tab")) {
    tab.classList.toggle("on", tab.dataset.view === ui.view.main);
    tab.classList.toggle("pick", tab.dataset.view === picker && tab.dataset.view !== ui.view.main);
  }
  const map = ui.view.main === "map";
  $("mode-switch").hidden = map;
  $("map-tools").hidden = !map;
  $("map-follow").classList.toggle("on", ui.mapView.follow);
  for (const button of document.querySelectorAll("#mode-switch button")) button.classList.toggle("on", button.dataset.mode === ui.view.mode);
  renderLegend();
  renderPickBanner();
}

function sendView() {
  send({ type: "view", main: ui.view.main, mode: ui.view.mode });
}

function renderPickBanner() {
  const banner = $("pick-banner");
  const onMap = ui.armed === "navigate_to";
  const onCamera = TARGET_SKILLS.includes(ui.armed);
  const ready = (onMap && ui.view.main === "map") || (onCamera && ui.view.main !== "map");
  $("stage").classList.toggle("picking", ready);
  banner.hidden = !(onMap || onCamera);
  if (onMap) banner.textContent = ui.view.main === "map" ? "Click the goal · drag to set a heading" : "Open the Map view to pick the goal";
  if (onCamera) banner.textContent = ui.view.main === "map" ? "Open a camera view to pick the point" : "Click the point in this camera view";
}

function renderLegend() {
  const legend = $("legend");
  legend.textContent = "";
  const row = (...children) => legend.append(el("div", { className: "row" }, ...children));
  if (ui.view.main === "map") {
    for (const [color, label] of [["#ffffff", "free"], ["#c8c8c8", "not seen yet"], ["#282828", "obstacle"], ["#fac8c8", "too close for the robot"]]) {
      row(el("span", { className: "sw", style: `background:${color}` }), el("span", { textContent: label }));
    }
    row(el("span", { className: "sw", style: "background:#f2c230" }), el("span", { textContent: "robot (true pose)" }));
    row(el("span", { className: "hint", textContent: "wheel: zoom · drag: pan" }));
  } else if (ui.view.mode === "depth") {
    const max = ui.hello ? ui.hello.max_depth : 6;
    row(el("span", { textContent: "0 m" }), el("span", { className: "ramp" }), el("span", { textContent: `${max} m` }));
    row(el("span", { textContent: "black: no depth" }));
  }
  legend.hidden = !legend.childElementCount;
}

function toast(text, state) {
  const node = el("div", { className: `toast s-${state}`, textContent: text });
  $("toasts").append(node);
  while ($("toasts").childElementCount > 3) $("toasts").firstChild.remove();
  setTimeout(() => node.remove(), 5000);
}

// ---------- skills ----------
function skillForTile(tile) {
  for (const skill of ui.skills.values()) if (skill.tile === tile) return skill;
  return null;
}

function buildSkills() {
  // Available skills as tiles by group, numbered for the keyboard; the others in one compact row.
  const root = $("skill-groups");
  root.textContent = "";
  const later = [];
  let number = 0;
  for (const [group, names] of SKILL_GROUPS) {
    const tiles = el("div", { className: "tiles" });
    for (const name of names) {
      const skill = skillForTile(name);
      if (!skill) { later.push(name); continue; }
      number += 1;
      const tile = el("button", { type: "button", className: "tile" },
        el("span", { className: "tile-top" }, icon(skill.name), el("span", { className: "name", textContent: name }), number <= 9 ? el("kbd", { textContent: String(number) }) : null),
        el("span", { className: "result", textContent: skill.immediate ? "runs at once" : "" }));
      tile.title = `${skill.call}\n${skill.doc}`;
      tile.dataset.skill = skill.name;
      tile.dataset.key = String(number);
      tile.addEventListener("click", () => selectSkill(skill.name));
      tiles.append(tile);
    }
    if (tiles.childElementCount) root.append(el("div", { className: "group" }, el("h3", { textContent: group }), tiles));
  }
  if (later.length) {
    const chips = el("div", { className: "later-chips" });
    for (const name of later) chips.append(el("span", { className: "later-chip", textContent: name, title: "Not available yet" }));
    root.append(el("div", { className: "group later" }, el("h3", { textContent: "COMING LATER" }), chips));
  }
  refreshTiles();
}

function latestOfSkill(name) {
  let latest = null;
  for (const record of ui.commands.values()) if (record.skill === name && (!latest || record.started >= latest.started)) latest = record;
  return latest;
}

function refreshTiles() {
  for (const tile of document.querySelectorAll(".tile[data-skill]")) {
    const name = tile.dataset.skill;
    const skill = ui.skills.get(name);
    const record = latestOfSkill(name);
    const result = tile.querySelector(".result");
    let state = "idle", text = skill && skill.immediate ? "runs at once" : "", resultState = "";
    if (record) {
      resultState = record.state;
      text = record.state === "failed" ? `✗ ${(record.feedback && record.feedback.failure) || record.status}` : `${SYMBOL[record.state] || ""} ${record.status}`;
      if (record.state === "executing") state = "executing";
    }
    if (ui.armed === name) { state = "armed"; resultState = "armed"; text = "◆ selected"; }
    tile.className = `tile${state === "idle" ? "" : ` s-${state}`}`;
    result.className = `result${resultState ? ` s-${resultState}` : ""}`;
    result.textContent = text;
  }
}

function selectSkill(name) {
  const skill = ui.skills.get(name);
  if (!skill) return;
  if (skill.immediate) { // No inputs: run on the tile.
    send({ type: "run", skill: name, inputs: {} });
    return;
  }
  ui.armed = ui.armed === name ? null : name;
  ui.goal = null;
  ui.target = null;
  ui.targetFound = null;
  ui.preview = null;
  ui.drive.keys.clear();
  if (ui.armed === "move_joints") setJointsFromState();
  if (ui.armed && skill.picker === "camera") { if (ui.view.main === "map") selectView("front"); else markTabs(); }
  else if (ui.armed && skill.picker && skill.picker !== ui.view.main) selectView(skill.picker);
  else if (ui.armed === "drive" && ui.view.main === "map") selectView("front");
  else markTabs();
  refreshTiles();
  renderSkillPanel();
}

function disarm() {
  ui.armed = null;
  ui.goal = null;
  ui.target = null;
  ui.targetFound = null;
  ui.preview = null;
  ui.drag = null;
  ui.drive.keys.clear();
  ui.drive.pad = [0, 0, 0];
  ui.drive.look = null;
  refreshTiles();
  renderSkillPanel();
  markTabs();
}

function runArmed() {
  if (ui.armed === "navigate_to") {
    if (!ui.goal) return;
    const inputs = { x: ui.goal.x, y: ui.goal.y, heading: ui.goal.heading };
    for (const field of document.querySelectorAll("#skill-panel input[data-input]")) inputs[field.dataset.input] = Number(field.value);
    send({ type: "run", skill: "navigate_to", inputs });
    disarm();
  } else if (ui.armed === "move_joints") {
    const duration = ui.jointDuration === "" ? null : Number(ui.jointDuration);
    send({ type: "run", skill: "move_joints", inputs: { positions: { ...ui.joints }, duration } });
  } else if (ui.armed === "execute_trajectory") {
    if (!ui.poses.length) return;
    send({ type: "run", skill: "execute_trajectory", inputs: { waypoints: ui.poses.map((p) => ({ positions: p.positions, seconds: p.seconds })) } });
  } else if (ui.armed && (TARGET_SKILLS.includes(ui.armed) || ui.armed === "move_along")) {
    const skill = ui.skills.get(ui.armed);
    if (TARGET_SKILLS.includes(skill.name) && !(ui.target && ui.targetFound)) return;
    const inputs = { ...choicesFor(skill) };
    if (TARGET_SKILLS.includes(skill.name)) inputs.target = { ...ui.target };
    send({ type: "run", skill: skill.name, inputs });
    disarm();
  }
}

// The current value of each non-target input: the person's choice, or the schema default.
function choicesFor(skill) {
  const chosen = ui.choices[skill.name] || {};
  const values = {};
  for (const input of skill.inputs) {
    if (input.name === "target") continue;
    values[input.name] = chosen[input.name] != null ? chosen[input.name] : input.default;
  }
  return values;
}

function setChoice(skill, name, value) {
  ui.choices[skill.name] = { ...(ui.choices[skill.name] || {}), [name]: value };
}

function renderSkillPanel() {
  const panel = $("skill-panel");
  panel.textContent = "";
  const skill = ui.armed && ui.skills.get(ui.armed);
  panel.classList.toggle("armed", Boolean(skill));
  $("skills-panel").classList.toggle("compact", Boolean(skill));
  if (!skill) {
    panel.append(
      el("div", { className: "panel-head" }, el("h2", { textContent: "Selected skill" }), el("span", { className: "pill s-idle", textContent: "none" })),
      el("div", { className: "doc", textContent: "Pick a skill above. Its inputs appear here, and the view switches to what it needs. Look at, Move to pose, Pick and Place take a point you click in a camera view. Stop, Stow, Open and Close run at once." }),
      keyList([["1–9", "select a skill"], ["Enter", "run the selected skill"], ["Esc", "clear the selection"], ["Pause", "E-stop: ends the episode"]]),
    );
    return;
  }
  panel.append(el("div", { className: "panel-head" },
    el("div", { className: "title" }, icon(skill.name), el("span", { textContent: skill.tile, title: skill.call })),
    el("span", { className: "pill s-armed", textContent: "selected" })));
  const last = latestOfSkill(skill.name);
  if (last) {
    const color = css({ succeeded: "--ok", failed: "--fail", executing: "--run" }[last.state] || "--muted");
    panel.append(el("div", { className: "last small mono", style: `color:${color}`, textContent: `last: ${resultText(last)}` }));
  }
  const renderers = {
    navigate_to: renderNavigate, drive: renderDrive, move_joints: renderJoints, execute_trajectory: renderPath,
  };
  (renderers[skill.name] || renderTargetSkill)(panel, skill);
}

function renderNavigate(panel, skill) {
  const goal = ui.goal;
  panel.append(el("div", {
    className: `prompt${goal ? " done" : ""}`,
    textContent: goal ? "Goal set. Click again to move it, or press Run." : "Click the goal on the map; drag from it to set a heading. The robot plans on what its cameras have seen.",
  }));
  const inputs = el("dl", { className: "inputs" });
  const row = (name, value, unit) => inputs.append(el("dt", { textContent: name }), el("dd", { textContent: value }), el("span", { className: "unit", textContent: unit }));
  row("x", goal ? fmt(goal.x) : "click map", goal ? "m" : "");
  row("y", goal ? fmt(goal.y) : "click map", goal ? "m" : "");
  row("heading", goal ? (goal.heading == null ? "any" : fmt(goal.heading * DEG, 0)) : "drag", goal && goal.heading != null ? "°" : "");
  const pose = ui.state && ui.state.pose;
  if (goal && pose) row("distance", fmt(Math.hypot(goal.x - pose.x, goal.y - pose.y)), "m");
  for (const input of skill.inputs.filter((i) => i.default != null)) {
    const id = `input-${input.name}`;
    const field = el("input", { type: "number", id, step: "0.01", min: input.min, max: input.max, value: input.default });
    field.dataset.input = input.name;
    inputs.append(el("dt", {}, el("label", { htmlFor: id, textContent: input.name.replace("_tolerance", " tol.") })), el("dd", {}, field), el("span", { className: "unit", textContent: input.unit }));
  }
  panel.append(inputs, actions([["Run  ⏎", runArmed, "primary", !goal], ["Clear  Esc", disarm, "", false]]));
}

function renderDrive(panel, skill) {
  const limits = skill.limits;
  const value = el("span", { className: "mono small" });
  const show = () => { value.textContent = `${fmt(ui.drive.fraction * limits.max_speed)} m/s · ${fmt(ui.drive.fraction * limits.max_yaw_speed)} rad/s`; };
  const slider = el("input", { type: "range", id: "drive-speed", min: 0.1, max: 1, step: 0.05, value: ui.drive.fraction });
  slider.addEventListener("input", () => { ui.drive.fraction = Number(slider.value); show(); });
  show();
  panel.append(
    el("div", { className: "prompt done", textContent: "Hold W/S, A/D, Q/E or a pad button. Releasing stops within 0.5 s." }),
    el("div", { className: "speed-row" }, el("label", { htmlFor: "drive-speed", className: "small muted", textContent: "Speed" }), value),
    slider, drivePad(),
    actions([[ui.drive.look ? `Stop looking (${fmt((ui.drive.look.turned * DEG), 0)}°)` : "Look around (one turn)", lookAround, "", false], ["Done  Esc", disarm, "", false]]),
  );
}

// Look around: turn on the spot once (unknown floor is blocked, so every run starts with it).
function lookAround() {
  const pose = ui.state && ui.state.pose;
  ui.drive.look = ui.drive.look || !pose ? null : { turned: 0, last: pose.yaw };
  renderSkillPanel();
}

function followLookAround(state) {
  const look = ui.drive.look;
  if (!look || !state.pose) return;
  look.turned += ((state.pose.yaw - look.last + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
  look.last = state.pose.yaw;
  if (Math.abs(look.turned) >= 2 * Math.PI) ui.drive.look = null; // The 0.5 s command runs out: the robot stops.
  if (ui.armed === "drive") renderSkillPanel();
}

// Look at, Move to pose, Move along: a clicked point and/or choices and numbers.
function renderTargetSkill(panel, skill) {
  const needsTarget = TARGET_SKILLS.includes(skill.name);
  if (needsTarget) {
    const found = ui.targetFound;
    let text = "Click the point in a camera view (Front, Hand, Front-left or Front-right).";
    if (ui.target && !found) text = "No depth at that pixel; click a surface the camera sees.";
    if (found && found.world) text = `Point (${found.world.map((v) => fmt(v)).join(", ")}) m · ${fmt(found.distance)} m from the camera${found.normal ? "" : " · no surface normal (edge)"}`;
    panel.append(el("div", { className: `prompt${found ? " done" : ""}`, textContent: text }));
  }
  const values = choicesFor(skill);
  const inputs = el("div", { className: "choices" });
  for (const input of skill.inputs) {
    if (input.name === "target") continue;
    const label = el("label", { className: "small muted", textContent: input.unit ? `${input.name.replace(/_(m|deg)$/, "")} [${input.unit}]` : input.name });
    if (input.choices) {
      const group = el("div", { className: "segmented choice", role: "group" });
      for (const choice of input.choices) {
        const button = el("button", { type: "button", textContent: choice, className: values[input.name] === choice ? "on" : "" });
        button.addEventListener("click", () => { setChoice(skill, input.name, choice); ui.preview = null; renderSkillPanel(); });
        group.append(button);
      }
      inputs.append(label, group);
    } else {
      const id = `choice-${skill.name}-${input.name}`;
      const value = el("span", { className: "mono small", textContent: fmt(values[input.name]) });
      const step = input.max - input.min > 10 ? 1 : 0.01;
      const slider = el("input", { type: "range", id, min: input.min, max: input.max, step, value: values[input.name] });
      slider.addEventListener("input", () => {
        setChoice(skill, input.name, Number(slider.value));
        value.textContent = fmt(Number(slider.value));
        if (ui.preview) { ui.preview = null; renderSkillPanel(); }
      });
      label.htmlFor = id;
      inputs.append(el("div", { className: "speed-row" }, label, value), slider);
    }
  }
  panel.append(inputs);
  const ready = !needsTarget || Boolean(ui.target && ui.targetFound);
  const plan = previewBlock(skill, () => ({ ...choicesFor(skill), target: { ...ui.target } }), ready);
  if (plan) panel.append(plan);
  panel.append(actions([["Run  ⏎", runArmed, "primary", !ready], ["Clear  Esc", disarm, "", false]]));
}

function measuredJoints() {
  return ui.state && ui.state.arm ? ui.state.arm.joints : null;
}

function setJointsFromState() {
  const measured = measuredJoints();
  ui.joints = Object.fromEntries(ARM.map((name) => [name, measured ? measured[name] : 0]));
}

function renderJoints(panel, skill) {
  if (!ui.joints) setJointsFromState();
  const limits = skill.limits || {};
  const list = el("div", { className: "joints" });
  for (const name of ARM) {
    const [lower, upper] = limits[name] || [-Math.PI, Math.PI];
    const id = `joint-${name}`;
    const slider = el("input", { type: "range", id, min: lower, max: upper, step: 0.005, value: ui.joints[name] });
    const value = el("span", { className: "mono value", id: `value-${name}` });
    const show = () => {
      value.textContent = fmt(ui.joints[name]);
      value.title = `${fmt(ui.joints[name] * DEG, 0)}°; limits ${fmt(lower)} to ${fmt(upper)} rad`;
    };
    const set = (q) => {
      ui.joints[name] = Math.min(upper, Math.max(lower, q));
      slider.value = ui.joints[name];
      show();
      if (ui.preview) { ui.preview = null; renderSkillPanel(); } // The plan was for other joints.
    };
    slider.addEventListener("input", () => set(Number(slider.value)));
    const minus = el("button", { type: "button", className: "nudge", textContent: "−", title: `−${NUDGE_RAD} rad` });
    const plus = el("button", { type: "button", className: "nudge", textContent: "+", title: `+${NUDGE_RAD} rad` });
    minus.addEventListener("click", () => set(ui.joints[name] - NUDGE_RAD));
    plus.addEventListener("click", () => set(ui.joints[name] + NUDGE_RAD));
    show();
    list.append(el("div", { className: "joint" },
      el("label", { htmlFor: id, textContent: JOINT_LABELS[name], title: name }), minus, slider, plus, value));
  }
  const duration = el("input", { type: "number", id: "joint-duration", min: 0.2, step: 0.1, placeholder: "shortest", value: ui.jointDuration });
  duration.addEventListener("input", () => { ui.jointDuration = duration.value; });
  const fromRobot = el("button", { type: "button", className: "link", textContent: "Reset to the robot's pose" });
  fromRobot.addEventListener("click", () => { setJointsFromState(); renderSkillPanel(); });
  panel.append(...[
    el("div", { className: "prompt done", textContent: "Set the joints [rad], then Go. The hand camera shows what the arm sees." }),
    list,
    el("div", { className: "small muted", id: "joints-now" }),
    el("div", { className: "speed-row" }, el("label", { htmlFor: "joint-duration", className: "small muted", textContent: "Duration [s]" }), duration),
    fromRobot,
    previewBlock(skill, () => ({ positions: { ...ui.joints } }), true),
    actions([["Go straight  ⏎", runArmed, "primary", false], ["Save pose", savePose, "", false]]),
  ].filter(Boolean));
  refreshJointNow();
}

function refreshJointNow() {
  const measured = measuredJoints();
  const node = $("joints-now");
  if (!measured || !node) return;
  node.textContent = `Robot now: ${ARM.map((name) => fmt(measured[name])).join("  ")}`;
}

function savePose() {
  ui.poses.push({ name: `Pose ${ui.poses.length + 1}`, positions: { ...ui.joints }, seconds: 3 });
  toast(`Saved Pose ${ui.poses.length}; play saved poses with Joint path`, "succeeded");
}

function renderPath(panel) {
  panel.append(el("div", {
    className: `prompt${ui.poses.length ? " done" : ""}`,
    textContent: ui.poses.length ? "The arm goes through these poses in order, from where it is now." : "No saved poses yet. Use Arm joints → Save pose.",
  }));
  const list = el("ol", { className: "poses" });
  ui.poses.forEach((pose, index) => {
    const seconds = el("input", { type: "number", id: `pose-seconds-${index}`, min: 0.5, step: 0.5, value: pose.seconds });
    seconds.addEventListener("input", () => { pose.seconds = Number(seconds.value) || pose.seconds; });
    const up = el("button", { type: "button", className: "nudge", textContent: "↑", disabled: index === 0, title: "Earlier" });
    const remove = el("button", { type: "button", className: "nudge", textContent: "×", title: "Remove" });
    up.addEventListener("click", () => { [ui.poses[index - 1], ui.poses[index]] = [ui.poses[index], ui.poses[index - 1]]; renderSkillPanel(); });
    remove.addEventListener("click", () => { ui.poses.splice(index, 1); renderSkillPanel(); });
    const summary = ARM.map((n) => fmt(pose.positions[n])).join("  ");
    list.append(el("li", {},
      el("div", { className: "pose-head" }, el("b", { textContent: `${index + 1}. ${pose.name}` }),
        el("label", { htmlFor: `pose-seconds-${index}`, className: "small muted", textContent: "after" }), seconds,
        el("span", { className: "small muted", textContent: "s" }), up, remove),
      el("div", { className: "pose-summary", textContent: summary, title: "sh0 sh1 el0 el1 wr0 wr1 [rad]" })));
  });
  panel.append(list, actions([["Run path  ⏎", runArmed, "primary", !ui.poses.length], ["Clear  Esc", disarm, "", false]]));
}

function keyList(rows) {
  const list = el("dl", { className: "keys small" });
  for (const [key, text] of rows) list.append(el("dt", {}, el("kbd", { textContent: key })), el("dd", { textContent: text }));
  return list;
}

function actions(list) {
  const row = el("div", { className: "actions" });
  for (const [text, action, cls, disabled] of list) {
    const button = el("button", { type: "button", textContent: text, className: cls, disabled });
    button.addEventListener("click", action);
    row.append(button);
  }
  return row;
}

function drivePad() {
  const pad = el("div", { className: "pad" });
  const keys = [["Q ⟲", [0, 0, 1]], ["W ▲", [1, 0, 0]], ["E ⟳", [0, 0, -1]], ["A ◀", [0, 1, 0]], ["S ▼", [-1, 0, 0]], ["D ▶", [0, -1, 0]]];
  for (const [text, direction] of keys) {
    const button = el("button", { type: "button", textContent: text });
    const press = (event) => { event.preventDefault(); ui.drive.pad = direction; button.classList.add("on"); };
    const release = () => { ui.drive.pad = [0, 0, 0]; button.classList.remove("on"); ui.drive.blocked = false; };
    button.addEventListener("pointerdown", press);
    button.addEventListener("pointerup", release);
    button.addEventListener("pointerleave", release);
    pad.append(button);
  }
  return pad;
}

// Drive: renew a 0.5 s velocity command every 0.2 s while a key or pad button is held.
function driveTick() {
  if (ui.armed !== "drive") return;
  const direction = [0, 0, 0];
  for (const key of ui.drive.keys) DRIVE_KEYS[key].forEach((v, i) => { direction[i] += v; });
  ui.drive.pad.forEach((v, i) => { direction[i] += v; });
  if (direction.some((v) => v !== 0)) ui.drive.look = null; // A key or the pad takes over.
  if (ui.drive.look) direction[2] = 1;
  if (direction.every((v) => v === 0)) { ui.drive.blocked = false; return; }
  if (ui.drive.blocked) return; // The last renewal was rejected: release the keys first.
  const limits = ui.skills.get("drive").limits;
  const f = ui.drive.look ? 1 : ui.drive.fraction; // Look around turns at full speed.
  send({
    type: "drive",
    v_x: Math.sign(direction[0]) * f * limits.max_speed,
    v_y: Math.sign(direction[1]) * f * limits.max_speed,
    v_rot: Math.sign(direction[2]) * f * limits.max_yaw_speed,
  });
}

// ---------- status cards and log ----------
function latestCommand(channel) {
  let latest = null;
  for (const record of ui.commands.values()) if (record.channel === channel && (!latest || record.started >= latest.started)) latest = record;
  return latest;
}

function activeCommand(channel) {
  const latest = latestCommand(channel);
  return latest && latest.state === "executing" ? latest : null;
}

function refreshCards() {
  for (const channel of ["base", "arm"]) {
    const card = $(`card-${channel}`);
    const record = latestCommand(channel);
    if (!record) continue;
    const now = simNow();
    const end = record.ended == null ? now : record.ended;
    const pill = card.querySelector(".pill");
    pill.className = `pill s-${record.state}`;
    pill.textContent = record.status;
    card.className = `panel channel s-${record.state}`;
    card.querySelector(".what").textContent = `${record.skill} #${record.id.replace(/^c/, "")}  ${inputsText(record)}`;
    const feedback = record.feedback || {};
    card.querySelector(".step").textContent = record.state === "failed" ? resultText(record) : record.step || feedback.reason || "";
    let meta = `started ${fmt(record.started, 1)} s · ${record.state === "executing" ? "running" : "took"} ${fmt(end - record.started, 1)} s`;
    const progress = card.querySelector(".progress");
    const pose = ui.state && ui.state.pose;
    if (record.skill === "navigate_to" && record.inputs && record.inputs.x != null && pose) {
      const left = Math.hypot(record.inputs.x - pose.x, record.inputs.y - pose.y);
      meta += ` · ${fmt(left)} m to the goal`;
      const done = record.startDistance ? 1 - left / record.startDistance : 0;
      progress.hidden = record.state !== "executing";
      progress.firstChild.style.width = `${Math.round(Math.min(1, Math.max(0, done)) * 100)}%`;
    } else {
      progress.hidden = true;
    }
    card.querySelector(".meta").textContent = meta;
    const cancel = card.querySelector(".cancel");
    cancel.hidden = record.state !== "executing";
    cancel.dataset.id = record.id;
  }
}

// Consecutive lines that differ only in their numbers collapse into one with a count.
function log(time, text, cls, channel, command) {
  const list = $("log");
  const shape = `${cls}|${channel}|${text.replace(/-?\d+(\.\d+)?/g, "#")}`;
  const last = list.lastElementChild;
  if (last && last.dataset.shape === shape) {
    const count = Number(last.dataset.count) + 1;
    last.dataset.count = String(count);
    if (command) last.dataset.command = command; // Click: the latest of the collapsed commands.
    last.querySelector(".t").textContent = fmt(time);
    last.querySelector(".text").textContent = text;
    last.querySelector(".count").textContent = `×${count}`;
    return;
  }
  const item = el("li", { className: cls || "" },
    el("span", { className: "t", textContent: fmt(time) }),
    el("span", { className: "text", textContent: text }),
    el("span", { className: "count" }));
  Object.assign(item.dataset, { channel, shape, count: "1" });
  if (command) item.dataset.command = command;
  item.hidden = !visibleInLog(item);
  const atBottom = list.scrollTop + list.clientHeight >= list.scrollHeight - 4;
  list.append(item);
  while (list.children.length > 800) list.firstChild.remove();
  if (atBottom) list.scrollTop = list.scrollHeight;
}

function visibleInLog(item) {
  if (ui.logFilter === "all") return true;
  if (ui.logFilter === "errors") return item.classList.contains("fail");
  return item.dataset.channel === ui.logFilter || item.dataset.channel === "all";
}

function simNow() {
  return ui.state ? ui.state.sim_time : 0;
}

// ---------- drawing ----------
function fitCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio)), height = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  return canvas.getContext("2d");
}

function drawImageFit(context, bitmap, width, height) {
  const scale = Math.min(width / bitmap.width, height / bitmap.height);
  const w = bitmap.width * scale, h = bitmap.height * scale;
  const x = (width - w) / 2, y = (height - h) / 2;
  context.drawImage(bitmap, x, y, w, h);
  return { x, y, scale };
}

function placeholder(context, width, height, text) {
  context.fillStyle = "#6b7782";
  context.font = `${Math.max(11, Math.min(18, width / 30))}px system-ui`;
  context.textAlign = "center";
  context.fillText(text, width / 2, height / 2);
  context.textAlign = "start";
}

function drawCamera(context, name, width, height, isMain) {
  context.fillStyle = "#000";
  context.fillRect(0, 0, width, height);
  const image = ui.images[name];
  if (!image) { placeholder(context, width, height, "waiting for frames"); return null; }
  const fit = drawImageFit(context, image.bitmap, width, height);
  if (isMain) {
    ui.imageFit = fit;
    const probe = ui.probe;
    if (ui.target && ui.target.view === name) { // The picked target: blue cross.
      const found = ui.targetFound;
      const [px, py] = found ? found.pixel : [ui.target.x, ui.target.y];
      const cx = fit.x + (px + 0.5) * fit.scale, cy = fit.y + (py + 0.5) * fit.scale;
      context.strokeStyle = "#2f7fd8";
      context.lineWidth = 3;
      context.beginPath(); context.arc(cx, cy, 10, 0, 2 * Math.PI); context.moveTo(cx - 16, cy); context.lineTo(cx + 16, cy);
      context.moveTo(cx, cy - 16); context.lineTo(cx, cy + 16); context.stroke();
    }
    if (name === "front") drawPreviewOnFront(context, image.header, fit);
    if (ui.cursor && probe && probe.view === name && probe.found) { // Mark the resolved pixel.
      const [px, py] = probe.found.pixel;
      const cx = fit.x + (px + 0.5) * fit.scale, cy = fit.y + (py + 0.5) * fit.scale;
      context.strokeStyle = "#f2c230";
      context.lineWidth = 2;
      context.beginPath(); context.arc(cx, cy, 6, 0, 2 * Math.PI); context.stroke();
    }
  }
  return image;
}

// The previewed plan on the Front view: body-frame points through its pinhole (x ahead, y left, z up),
// tilted down by header.pitch [rad].
function drawPreviewOnFront(context, header, fit) {
  const preview = ui.preview;
  if (!preview || preview.status !== "ready" || ui.armed !== preview.skill || !header.origin) return;
  const [ox, oy, oz] = header.origin;
  const c = Math.cos(header.pitch || 0), s = Math.sin(header.pitch || 0);
  const toCanvas = ([x, y, z]) => {
    const ahead = c * (x - ox) - s * (z - oz), up = s * (x - ox) + c * (z - oz);  // The view's axes.
    if (ahead < 0.05) return null;
    const col = header.width / 2 - ((y - oy) * header.focal) / ahead, row = header.height / 2 - (up * header.focal) / ahead;
    return [fit.x + col * fit.scale, fit.y + row * fit.scale];
  };
  const line = (points, color, width, dash) => {
    context.strokeStyle = color;
    context.lineWidth = width;
    context.setLineDash(dash);
    context.beginPath();
    let drawing = false;
    for (const point of points.map(toCanvas)) {
      if (!point) { drawing = false; continue; }
      if (drawing) context.lineTo(...point); else context.moveTo(...point);
      drawing = true;
    }
    context.stroke();
    context.setLineDash([]);
  };
  line(preview.arm_body, "rgba(94,164,242,0.9)", 5, []); // The arm where the plan ends.
  for (const point of preview.arm_body.map(toCanvas)) {
    if (!point) continue;
    context.fillStyle = "#5ea4f2";
    context.beginPath(); context.arc(point[0], point[1], 4, 0, 2 * Math.PI); context.fill();
  }
  line(preview.path_body, "#f2c230", 3, [7, 5]); // The gripper's path.
  const end = toCanvas(preview.path_body[preview.path_body.length - 1]);
  if (end) { context.strokeStyle = "#f2c230"; context.lineWidth = 3; context.beginPath(); context.arc(end[0], end[1], 8, 0, 2 * Math.PI); context.stroke(); }
}

function mapLayout(width, height, isMain) {
  const header = ui.map.header;
  const base = Math.min(width / header.width, height / header.height);
  const res = header.resolution;
  let zoom = 1, center = [header.origin[0] + (header.width * res) / 2, header.origin[1] + (header.height * res) / 2];
  if (isMain) {
    const pose = ui.state && ui.state.pose;
    if (ui.mapView.follow && pose) center = [pose.x, pose.y];
    else if (ui.mapView.center) center = ui.mapView.center;
    zoom = ui.mapView.zoom;
  }
  const scale = base * zoom;
  const col = (center[0] - header.origin[0]) / res, row = header.height - (center[1] - header.origin[1]) / res;
  return { x: width / 2 - col * scale, y: height / 2 - row * scale, scale };
}

function drawMap(context, width, height, isMain) {
  context.fillStyle = "#1a1f24";
  context.fillRect(0, 0, width, height);
  if (!ui.map) { placeholder(context, width, height, "no map yet"); return; }
  const header = ui.map.header;
  const fit = mapLayout(width, height, isMain);
  context.imageSmoothingEnabled = false;
  context.drawImage(ui.map.bitmap, fit.x, fit.y, header.width * fit.scale, header.height * fit.scale);
  context.imageSmoothingEnabled = true;
  if (isMain) ui.mapFit = fit;
  const perMetre = fit.scale / header.resolution;
  const toCanvas = (wx, wy) => [
    fit.x + ((wx - header.origin[0]) / header.resolution) * fit.scale,
    fit.y + (header.height - (wy - header.origin[1]) / header.resolution) * fit.scale,
  ];
  if (ui.trail.length > 1) {
    context.strokeStyle = "#3d8fe0";
    context.lineWidth = Math.max(1.5, perMetre * 0.03);
    context.beginPath();
    ui.trail.forEach(([wx, wy], i) => { const [cx, cy] = toCanvas(wx, wy); if (i) context.lineTo(cx, cy); else context.moveTo(cx, cy); });
    context.stroke();
  }
  const pose = ui.state && ui.state.pose;
  const goals = [];
  if (ui.lastGoal) {
    const record = ui.commands.get(ui.lastGoal.id);
    const state = record ? record.state : "executing";
    goals.push([ui.lastGoal, { executing: "#e0a020", succeeded: "#1d9a5f", failed: "#d0304f", cancelled: "#7c8894" }[state], state === "executing"]);
  }
  if (ui.goal) goals.push([ui.goal, "#2f7fd8", true]);
  for (const [goal, color, line] of goals) {
    if (line && pose) { // A straight guide from the robot: not the planned path, which the API does not return.
      const [ax, ay] = toCanvas(pose.x, pose.y), [bx, by] = toCanvas(goal.x, goal.y);
      context.strokeStyle = color;
      context.lineWidth = 1.5;
      context.setLineDash([6, 5]);
      context.beginPath(); context.moveTo(ax, ay); context.lineTo(bx, by); context.stroke();
      context.setLineDash([]);
    }
    drawGoal(context, toCanvas, goal, color, perMetre);
  }
  const preview = ui.preview;
  if (preview && preview.status === "ready" && ui.armed === preview.skill) { // The previewed gripper path.
    context.strokeStyle = "#f2c230";
    context.lineWidth = 2.5;
    context.setLineDash([6, 4]);
    context.beginPath();
    preview.path_world.forEach(([wx, wy], i) => { const [cx, cy] = toCanvas(wx, wy); if (i) context.lineTo(cx, cy); else context.moveTo(cx, cy); });
    context.stroke();
    context.setLineDash([]);
  }
  if (pose) {
    const [cx, cy] = toCanvas(pose.x, pose.y);
    const size = Math.max(isMain ? 9 : 5, 0.55 * perMetre); // About half of Spot's length.
    context.save();
    context.translate(cx, cy);
    context.rotate(-pose.yaw);
    context.fillStyle = "#f2c230";
    context.strokeStyle = "#1d1600";
    context.lineWidth = 1.5;
    context.beginPath();
    context.moveTo(size, 0); context.lineTo(-size * 0.7, size * 0.55); context.lineTo(-size * 0.7, -size * 0.55); context.closePath();
    context.fill(); context.stroke();
    context.restore();
  }
  if (isMain) drawScaleBar(context, width, height, perMetre);
}

function drawGoal(context, toCanvas, goal, color, perMetre) {
  const [cx, cy] = toCanvas(goal.x, goal.y);
  const radius = Math.max(7, 0.15 * perMetre);
  context.strokeStyle = color;
  context.lineWidth = 3;
  context.beginPath(); context.arc(cx, cy, radius, 0, 2 * Math.PI); context.stroke();
  context.beginPath(); context.moveTo(cx - radius * 0.5, cy); context.lineTo(cx + radius * 0.5, cy);
  context.moveTo(cx, cy - radius * 0.5); context.lineTo(cx, cy + radius * 0.5); context.stroke();
  if (goal.heading != null) {
    const length = Math.max(22, 0.5 * perMetre);
    context.beginPath(); context.moveTo(cx, cy); context.lineTo(cx + Math.cos(goal.heading) * length, cy - Math.sin(goal.heading) * length); context.stroke();
  }
}

function drawScaleBar(context, width, height, perMetre) {
  const ratio = window.devicePixelRatio || 1;
  const metres = [0.25, 0.5, 1, 2, 5].find((m) => m * perMetre >= 60 * ratio) || 5;
  const length = metres * perMetre;
  const x = width / 2 - length / 2, y = height - 18 * ratio;
  context.fillStyle = "rgba(10,14,18,0.78)";
  context.fillRect(x - 8 * ratio, y - 16 * ratio, length + 16 * ratio, 24 * ratio);
  context.strokeStyle = "#e4e9ee";
  context.lineWidth = 2 * ratio;
  context.beginPath();
  context.moveTo(x, y - 4 * ratio); context.lineTo(x, y); context.lineTo(x + length, y); context.lineTo(x + length, y - 4 * ratio);
  context.stroke();
  context.fillStyle = "#e4e9ee";
  context.font = `${11 * ratio}px system-ui`;
  context.textAlign = "center";
  context.fillText(`${metres} m · north up`, x + length / 2, y - 5 * ratio);
  context.textAlign = "start";
}

function canvasPoint(event) {
  const canvas = $("main-view");
  const ratio = canvas.width / canvas.clientWidth;
  return [event.offsetX * ratio, event.offsetY * ratio];
}

function canvasToWorld(event) {
  if (!ui.map || !ui.mapFit) return null;
  const [px, py] = canvasPoint(event);
  const { x, y, scale } = ui.mapFit, header = ui.map.header;
  const col = (px - x) / scale, row = (py - y) / scale;
  return [header.origin[0] + col * header.resolution, header.origin[1] + (header.height - row) * header.resolution];
}

function canvasToPixel(event) {
  const image = ui.images[ui.view.main];
  if (!image || !ui.imageFit) return null;
  const [px, py] = canvasPoint(event);
  const x = Math.floor((px - ui.imageFit.x) / ui.imageFit.scale), y = Math.floor((py - ui.imageFit.y) / ui.imageFit.scale);
  if (x < 0 || y < 0 || x >= image.header.width || y >= image.header.height) return null;
  return { x, y, tick: image.header.tick };
}

function drawViewChip(image) {
  const chip = $("view-chip");
  if (ui.view.main === "map") {
    chip.textContent = ui.map ? `Map · ${ui.map.header.resolution} m cells · ×${fmt(ui.mapView.zoom, 1)}` : "Map";
    chip.title = "Built from the front cameras' depth; unknown space is blocked for navigation";
    chip.classList.remove("stale");
    return;
  }
  if (!image) { chip.textContent = VIEW_LABELS[ui.view.main]; return; }
  const waiting = ui.state && ui.state.phase !== "running"; // Not started, or over: no new frames.
  const ageWall = waiting ? 0 : (performance.now() - image.receivedAt) / 1000;
  const ageSim = simNow() - image.header.time;
  chip.textContent = `${VIEW_LABELS[image.header.view]} · ${image.header.width}×${image.header.height} · frame ${fmt(image.header.time)} s · ${fmt(Math.max(0, ageSim))} s old${ageWall > 1.5 ? " · STALE" : ""}`;
  chip.classList.toggle("stale", ageWall > 1.5);
}

function drawCursorChip() {
  const chip = $("cursor-chip");
  if (!ui.cursor) { chip.hidden = true; return; }
  chip.hidden = false;
  const pose = ui.state && ui.state.pose;
  if (ui.view.main === "map") {
    const [x, y] = ui.cursor;
    chip.textContent = `cursor ${fmt(x)}, ${fmt(y)} m${pose ? ` · ${fmt(Math.hypot(x - pose.x, y - pose.y))} m from the robot` : ""}`;
    return;
  }
  const probe = ui.probe;
  const pixel = ui.cursor;
  if (!probe || probe.view !== ui.view.main) { chip.textContent = `pixel ${pixel.x}, ${pixel.y}`; return; }
  if (!probe.found) { chip.textContent = `pixel ${pixel.x}, ${pixel.y} · no depth here`; return; }
  const found = probe.found;
  const where = found.world ? `world (${found.world.map((v) => fmt(v)).join(", ")}) m` : `body (${found.body.map((v) => fmt(v)).join(", ")}) m`;
  chip.textContent = `pixel ${found.pixel[0]}, ${found.pixel[1]} · ${fmt(found.distance)} m away · ${where}`;
}

function drawTimeline() {
  const canvas = $("timeline");
  const context = fitCanvas(canvas);
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.width;
  context.clearRect(0, 0, width, canvas.height);
  const now = simNow();
  const span = Math.max(60, now * 1.05);
  const left = 64 * ratio, right = width - 8 * ratio;
  const toX = (t) => left + (t / span) * (right - left);
  const lane = 20 * ratio, lanes = { base: 4 * ratio, arm: 28 * ratio };
  context.font = `${11 * ratio}px system-ui`;
  context.fillStyle = css("--muted");
  context.fillText("BASE", 4 * ratio, lanes.base + 14 * ratio);
  context.fillText("ARM", 4 * ratio, lanes.arm + 14 * ratio);
  context.fillStyle = css("--raise");
  for (const y of Object.values(lanes)) context.fillRect(left, y, right - left, lane);
  const fills = { executing: "--run-bg", succeeded: "--ok-bg", failed: "--fail-bg", cancelled: "--raise" };
  const edges = { executing: "--run", succeeded: "--ok", failed: "--fail", cancelled: "--idle" };
  ui.timelineHits = [];
  for (const record of ui.commands.values()) {
    const y = lanes[record.channel];
    if (y == null) continue;
    const x0 = toX(record.started), x1 = Math.max(x0 + 4 * ratio, toX(record.ended == null ? now : record.ended));
    context.fillStyle = css(fills[record.state] || "--raise");
    context.strokeStyle = css(edges[record.state] || "--idle");
    context.lineWidth = 1.5 * ratio;
    context.setLineDash(record.state === "cancelled" ? [3 * ratio, 2 * ratio] : []);
    context.fillRect(x0, y, x1 - x0, lane);
    context.strokeRect(x0, y, x1 - x0, lane);
    ui.timelineHits.push([x0 / ratio, y / ratio, x1 / ratio, (y + lane) / ratio, record]);
    if (x1 - x0 > 60 * ratio) {
      context.fillStyle = css("--fg");
      context.fillText(`${SYMBOL[record.state] || ""} ${record.skill}`, x0 + 5 * ratio, y + 14 * ratio);
    }
  }
  context.setLineDash([]);
  context.fillStyle = css("--muted");
  const step = span > 600 ? 120 : span > 300 ? 60 : span > 120 ? 30 : 10;
  for (let t = 0; t <= span; t += step) {
    context.fillRect(toX(t), 50 * ratio, 1, 4 * ratio);
    if (toX(t) + 40 * ratio < right) context.fillText(`${t} s`, toX(t) + 3 * ratio, 64 * ratio);
  }
  context.fillStyle = css("--spot");
  context.fillRect(toX(now) - ratio, 0, 2 * ratio, 54 * ratio);
}

function render() {
  const main = $("main-view");
  const context = fitCanvas(main);
  let image = null;
  if (ui.view.main === "map") drawMap(context, main.width, main.height, true);
  else image = drawCamera(context, ui.view.main, main.width, main.height, true);
  drawViewChip(image);
  drawCursorChip();
  const pip = $("pip");
  const pipContext = fitCanvas(pip);
  if (ui.view.main === "map") drawCamera(pipContext, "front", pip.width, pip.height, false);
  else drawMap(pipContext, pip.width, pip.height, false);
  for (const tab of document.querySelectorAll(".tab")) {
    const canvas = tab.querySelector("canvas");
    const thumb = canvas.getContext("2d");
    if (tab.dataset.view === "map") drawMap(thumb, canvas.width, canvas.height, false);
    else drawCamera(thumb, tab.dataset.view, canvas.width, canvas.height, false);
  }
  drawTimeline();
  requestAnimationFrame(render);
}

function formatWall(seconds) {
  const s = Math.floor(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

// ---------- input ----------
function estop() {
  send({ type: "estop" });
}

function typing(event) {
  return event.target instanceof HTMLInputElement && event.target.type !== "range";
}

function currentMapCenter() {
  const header = ui.map.header, res = header.resolution;
  const pose = ui.state && ui.state.pose;
  if (ui.mapView.follow && pose) return [pose.x, pose.y];
  return ui.mapView.center || [header.origin[0] + (header.width * res) / 2, header.origin[1] + (header.height * res) / 2];
}

function fitMap() {
  ui.mapView = { zoom: 1, center: null, follow: false };
  markTabs();
}

function setupMainView() {
  const main = $("main-view");
  main.addEventListener("contextmenu", (event) => event.preventDefault());
  main.addEventListener("pointerdown", (event) => {
    if (ui.view.main !== "map" && TARGET_SKILLS.includes(ui.armed) && event.button === 0) {
      const pixel = canvasToPixel(event);
      if (!pixel) return;
      ui.target = { view: ui.view.main, ...pixel };
      ui.targetFound = null;
      ui.preview = null;
      send({ type: "probe", view: ui.target.view, tick: ui.target.tick, x: ui.target.x, y: ui.target.y });
      renderSkillPanel();
      return;
    }
    if (ui.view.main !== "map" || !ui.map) return;
    const world = canvasToWorld(event);
    if (!world) return;
    main.setPointerCapture(event.pointerId);
    if (ui.armed === "navigate_to" && event.button === 0) {
      ui.drag = { x: world[0], y: world[1] };
      ui.goal = { x: world[0], y: world[1], heading: null };
      renderSkillPanel();
    } else {
      ui.pan = { point: canvasPoint(event), center: currentMapCenter() };
      ui.mapView.follow = false;
      markTabs();
    }
  });
  main.addEventListener("pointermove", (event) => {
    if (ui.view.main === "map") {
      ui.cursor = canvasToWorld(event);
      if (ui.pan && ui.mapFit) {
        const [px, py] = canvasPoint(event);
        const res = ui.map.header.resolution / ui.mapFit.scale;
        ui.mapView.center = [ui.pan.center[0] - (px - ui.pan.point[0]) * res, ui.pan.center[1] + (py - ui.pan.point[1]) * res];
      }
      if (ui.drag && ui.cursor) {
        const dx = ui.cursor[0] - ui.drag.x, dy = ui.cursor[1] - ui.drag.y;
        ui.goal.heading = Math.hypot(dx, dy) >= MIN_HEADING_DRAG_M ? Math.atan2(dy, dx) : null;
      }
      return;
    }
    ui.cursor = canvasToPixel(event);
    const now = performance.now();
    if (ui.cursor && now - ui.probeSent > 100) { // Ask what this pixel is, at most 10 times a second.
      ui.probeSent = now;
      send({ type: "probe", view: ui.view.main, tick: ui.cursor.tick, x: ui.cursor.x, y: ui.cursor.y });
    }
  });
  main.addEventListener("pointerup", () => {
    if (ui.drag) { ui.drag = null; renderSkillPanel(); }
    ui.pan = null;
  });
  main.addEventListener("pointerleave", () => { ui.cursor = null; });
  main.addEventListener("wheel", (event) => {
    if (ui.view.main !== "map" || !ui.map) return;
    event.preventDefault();
    const world = canvasToWorld(event);
    const center = currentMapCenter();
    const factor = event.deltaY < 0 ? 1.25 : 1 / 1.25;
    const zoom = Math.min(16, Math.max(1, ui.mapView.zoom * factor));
    const applied = zoom / ui.mapView.zoom;
    ui.mapView.zoom = zoom;
    if (world && !ui.mapView.follow) ui.mapView.center = [world[0] + (center[0] - world[0]) / applied, world[1] + (center[1] - world[1]) / applied];
  }, { passive: false });
  main.addEventListener("dblclick", () => { if (ui.view.main === "map") fitMap(); });
}

// ---------- telemetry: a drawer, closed by default ----------
const TELEMETRY_WINDOW_S = 60; // Simulated time the charts show [s].
const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"];

function recordTelemetry(state) {
  const t = state.telemetry;
  if (!t) return;
  const history = ui.telemetry.history;
  if (history.length && state.sim_time < history[history.length - 1].t) history.length = 0; // A replay restarted.
  history.push({ t: state.sim_time, rtf: state.rtf, base: t.base, speeds: t.joint_speeds, tracking: t.tracking });
  while (history.length && history[0].t < state.sim_time - TELEMETRY_WINDOW_S) history.shift();
  ui.telemetry.latest = state;
  if (!$("telemetry").hidden && performance.now() - ui.telemetry.drawn > 250) renderTelemetry();
}

function toggleTelemetry(open) {
  const drawer = $("telemetry");
  drawer.hidden = open == null ? !drawer.hidden : !open;
  $("telemetry-toggle").classList.toggle("on", !drawer.hidden);
  if (!drawer.hidden) renderTelemetry();
}

function inspectCommand(id) {
  ui.telemetry.inspect = id;
  ui.telemetry.tab = "command";
  toggleTelemetry(true);
}

function renderTelemetry() {
  ui.telemetry.drawn = performance.now();
  for (const button of document.querySelectorAll("#telemetry-tabs button")) button.classList.toggle("on", button.dataset.tab === ui.telemetry.tab);
  const body = $("telemetry-body");
  const scroll = body.scrollTop;
  body.textContent = "";
  const state = ui.telemetry.latest;
  const tabs = { health: telemetryHealth, base: telemetryBase, arm: telemetryArm, mission: telemetryMission, command: telemetryCommand };
  if (!state && ui.telemetry.tab !== "command") body.append(el("p", { className: "small muted", textContent: "Waiting for the robot's state." }));
  else tabs[ui.telemetry.tab](body, state);
  body.scrollTop = scroll;
}

function section(title, ...children) {
  return el("section", { className: "tm-section" }, el("h3", { textContent: title }), ...children);
}

// Rows of [label, value, level]; level ok, warn or bad colours a dot, null shows none.
function rows(list) {
  const dl = el("dl", { className: "tm-rows" });
  for (const [label, value, level] of list) {
    dl.append(el("dt", { textContent: label }), el("dd", { textContent: value }), el("span", { className: level ? `tm-dot ${level}` : "" }));
  }
  return dl;
}

// A strip chart of series [{label, color, value(entry)}] over the history, in its own scale.
function chart(series, unit) {
  const canvas = el("canvas", { className: "tm-chart" });
  const legend = el("div", { className: "tm-legend" });
  for (const line of series) {
    const item = el("span", { textContent: line.label });
    item.style.setProperty("--swatch", css(line.color));
    legend.append(item);
  }
  requestAnimationFrame(() => drawChart(canvas, series, unit));
  return el("div", {}, canvas, legend);
}

function drawChart(canvas, series, unit) {
  const context = fitCanvas(canvas);
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.width, height = canvas.height, history = ui.telemetry.history;
  context.clearRect(0, 0, width, height);
  if (history.length < 2) return;
  const now = history[history.length - 1].t, start = now - TELEMETRY_WINDOW_S;
  let low = 0, high = 0;
  for (const line of series) for (const entry of history) {
    const value = line.value(entry);
    if (value == null || !Number.isFinite(value)) continue;
    low = Math.min(low, value); high = Math.max(high, value);
  }
  if (high - low < 1e-6) { high += 1; low -= 1; }
  const pad = 0.08 * (high - low);
  low -= pad; high += pad;
  const left = 46 * ratio, right = width - 6 * ratio, top = 6 * ratio, bottom = height - 6 * ratio;
  const x = (t) => left + ((t - start) / TELEMETRY_WINDOW_S) * (right - left);
  const y = (v) => bottom - ((v - low) / (high - low)) * (bottom - top);
  context.font = `${10 * ratio}px system-ui`;
  context.fillStyle = css("--muted");
  context.strokeStyle = css("--line");
  context.lineWidth = ratio;
  for (const value of [high - pad, 0, low + pad]) {
    context.beginPath(); context.moveTo(left, y(value)); context.lineTo(right, y(value)); context.stroke();
    context.fillText(fmt(value, Math.abs(value) < 10 ? 2 : 0), 4 * ratio, y(value) + 3 * ratio);
  }
  for (const line of series) {
    context.strokeStyle = css(line.color);
    context.lineWidth = 1.6 * ratio;
    context.beginPath();
    let drawing = false;
    for (const entry of history) {
      const value = line.value(entry);
      if (value == null || !Number.isFinite(value)) { drawing = false; continue; }
      if (drawing) context.lineTo(x(entry.t), y(value)); else context.moveTo(x(entry.t), y(value));
      drawing = true;
    }
    context.stroke();
  }
}

function telemetryHealth(body, state) {
  const t = state.telemetry || {}, health = t.health || {}, link = ui.telemetry.rate;
  const rtf = state.rtf;
  const level = (value, warn, bad, higherIsBetter = false) => value == null ? null
    : higherIsBetter ? (value >= warn ? "ok" : value >= bad ? "warn" : "bad") : (value <= warn ? "ok" : value <= bad ? "warn" : "bad");
  const list = [
    ["Speed vs real time", rtf == null ? "–" : `×${fmt(rtf, 2)}`, level(rtf, 0.4, 0.2, true)],
    ["Physics per loop", health.physics_ms == null ? "–" : `${fmt(health.physics_ms, 0)} ms`, null],
    ["UI per loop", health.ui_ms == null ? "–" : `${fmt(health.ui_ms, 0)} ms`, level(health.ui_ms, 30, 80)],
  ];
  const map = health.pipeline && health.pipeline.map;
  if (map) list.push(["Map update", `${fmt(map.mean_ms, 0)} ms mean · ${fmt(map.max_ms, 0)} max`, level(map.mean_ms, 150, 300)]);
  const planner = health.planner;
  if (planner) {
    list.push(["Arm planner", planner.running ? (planner.busy ? "planning" : "ready") : "not running", planner.running ? "ok" : "bad"]);
    list.push(["Planner restarts", String(planner.restarts), null]);
    const last = planner.last_plan;
    if (last) list.push(["Last plan", `${last.kind} · ${fmt(last.seconds, 1)} s · ${last.result}`, last.result === "planned" ? "ok" : "warn"]);
  }
  body.append(section("Simulation and services", rows(list)));
  const cameras = Object.entries(t.cameras || {}).map(([name, camera]) => [
    VIEW_LABELS[name] || name,
    `${camera.fps == null ? "–" : fmt(camera.fps, 1)} Hz · ${fmt(camera.age, 2)} s old`,
    level(camera.age, 0.1, 0.5),
  ]);
  body.append(section("Cameras (simulated time)", rows(cameras)));
  body.append(section("Link to this page", rows([
    ["Received", link ? `${fmt(link.bytes / 1024, 0)} kB/s` : "–", null],
    ["Images", link ? `${link.frames}/s` : "–", null],
  ])));
  body.append(section("Speed vs real time", chart([{ label: "×RTF", color: "--s1", value: (e) => e.rtf }], "")));
}

function telemetryBase(body, state) {
  const base = state.telemetry && state.telemetry.base, pose = state.pose;
  body.append(section("Now", rows([
    ["Position", pose ? `${fmt(pose.x)}, ${fmt(pose.y)} m` : "–", null],
    ["Heading", pose ? `${fmt(pose.yaw * DEG, 0)}°` : "–", null],
    ["Forward · left", base ? `${fmt(base.vx)} · ${fmt(base.vy)} m/s` : "–", null],
    ["Turn rate", base && base.yaw_rate != null ? `${fmt(base.yaw_rate)} rad/s` : "–", null],
  ])));
  body.append(section("Velocity, body frame [m/s]", chart([
    { label: "forward", color: "--s1", value: (e) => e.base && e.base.vx },
    { label: "left", color: "--s2", value: (e) => e.base && e.base.vy },
  ], "m/s")));
  body.append(section("Turn rate [rad/s]", chart([{ label: "yaw rate", color: "--s3", value: (e) => e.base && e.base.yaw_rate }], "rad/s")));
}

function telemetryArm(body, state) {
  const t = state.telemetry || {}, arm = state.arm;
  const skill = ui.skills.get("move_joints");
  const limits = (skill && skill.limits) || {};
  const bars = el("div", { style: "display:grid;gap:5px" });
  for (const name of ARM) {
    const value = arm && arm.joints ? arm.joints[name] : null;
    const [lower, upper] = limits[name] || [-Math.PI, Math.PI];
    const margin = value == null ? null : Math.min(value - lower, upper - value);
    const mark = el("span", { className: "mark" });
    if (value != null) mark.style.left = `${Math.min(100, Math.max(0, ((value - lower) / (upper - lower)) * 100))}%`;
    const row = el("div", { className: `tm-limit${margin != null && margin < 0.1 ? " near" : ""}` },
      el("span", { className: "muted", textContent: JOINT_LABELS[name] }),
      el("span", { className: "track", title: `${fmt(lower)} to ${fmt(upper)} rad` }, mark),
      el("span", { className: "margin", textContent: margin == null ? "–" : `${fmt(margin * DEG, 0)}°`, title: "Margin to the nearer limit" }));
    bars.append(row);
  }
  body.append(section("Joints within their limits (margin to the nearer one)", bars));
  const hand = t.hand, tracking = t.tracking;
  const list = [
    ["Hand (world)", hand ? `${hand.map((v) => fmt(v)).join(", ")} m` : "–", null],
    ["Gripper", arm && arm.gripper != null ? `${fmt(arm.gripper)} rad` : "–", null],
    ["Holding", state.holding == null ? "–" : state.holding ? "yes" : "no", null],
  ];
  if (tracking) {
    list.push(["Running", `${tracking.skill} #${tracking.id.replace(/^c/, "")}`, "warn"]);
    if (tracking.progress != null) list.push(["Progress", `${fmt(tracking.progress * 100, 0)} %`, null]);
    if (tracking.joint_error != null) list.push(["Joint error", `${fmt(tracking.joint_error, 3)} rad`, null]);
    if (Array.isArray(tracking.pose_error)) list.push(["Pose error", `${fmt(tracking.pose_error[0], 3)} m · ${fmt(tracking.pose_error[1], 3)} rad`, null]);
  }
  body.append(section("Hand and command", rows(list)));
  body.append(section("Joint speeds [rad/s]", chart(ARM.map((name, i) => ({
    label: JOINT_LABELS[name], color: SERIES[i], value: (e) => e.speeds && e.speeds[name],
  })), "rad/s")));
}

function telemetryMission(body, state) {
  const t = state.telemetry || {}, task = state.task;
  if (task && task.budget != null) {
    const left = Math.max(0, task.budget - task.elapsed);
    const bar = el("div", { className: `tm-bar${left < 0.1 * task.budget ? " low" : ""}` }, el("span", { style: `width:${Math.min(100, (left / task.budget) * 100)}%` }));
    body.append(section("Budget", bar, rows([
      ["Left", `${fmt(left, 0)} of ${fmt(task.budget, 0)} s`, null],
      ["Wall time at this speed", state.rtf ? `≈${formatWall(left / state.rtf)}` : "–", null],
    ])));
  }
  const counts = Object.entries(t.commands || {}).map(([skill, byState]) => [
    skill, Object.entries(byState).map(([name, count]) => `${count} ${name}`).join(" · "), byState.failed ? "warn" : "ok",
  ]);
  body.append(section("Commands that ended", counts.length ? rows(counts) : el("p", { className: "small muted", textContent: "None yet." })));
  const failures = [];
  for (const [skill, codes] of Object.entries(t.failures || {})) for (const [code, count] of Object.entries(codes)) failures.push([`${skill}`, `${code} ×${count}`, "bad"]);
  body.append(section("Failures by code", failures.length ? rows(failures) : el("p", { className: "small muted", textContent: "None." })));
}

function telemetryCommand(body) {
  // The latest commands, newest first: pick one here, in the log or on the timeline.
  const recent = [...ui.commands.values()].sort((a, b) => b.started - a.started).slice(0, 8);
  const list = el("div", { className: "tm-commands" });
  for (const item of recent) {
    const button = el("button", { type: "button", className: item.id === ui.telemetry.inspect ? "on" : "" },
      el("span", { textContent: `${SYMBOL[item.state] || ""} ${item.skill} #${item.id.replace(/^c/, "")}` }),
      el("span", { className: "muted", textContent: `${fmt(item.started, 1)} s` }));
    button.addEventListener("click", () => { ui.telemetry.inspect = item.id; renderTelemetry(); });
    list.append(button);
  }
  body.append(section("Commands", recent.length ? list : el("p", { className: "small muted", textContent: "None yet." })));
  const record = ui.telemetry.inspect && ui.commands.get(ui.telemetry.inspect);
  if (!record) {
    body.append(el("p", { className: "small muted", textContent: "Pick a command to see its inputs and the API's raw feedback." }));
    return;
  }
  const { skill, id, channel, status, state, started, ended, inputs, feedback, step } = record;
  body.append(section(`${skill} #${id.replace(/^c/, "")}`, rows([
    ["State", `${state} (${status})`, { succeeded: "ok", failed: "bad", executing: "warn" }[state] || null],
    ["Channel", channel, null],
    ["Time", `${fmt(started, 1)}–${ended == null ? "…" : fmt(ended, 1)} s`, null],
    ["Last step", step || "–", null],
  ])));
  body.append(section("Inputs", el("pre", { textContent: JSON.stringify(inputs, null, 2) })));
  body.append(section("Feedback (raw)", el("pre", { textContent: JSON.stringify(feedback, null, 2) })));
}

function setup() {
  $("estop").addEventListener("click", estop);
  $("end-episode").addEventListener("click", endEpisode);
  $("start-button").addEventListener("click", startEpisode);
  $("result-close").addEventListener("click", () => { ui.resultClosed = true; $("result-card").hidden = true; });
  $("replay-play").addEventListener("click", () => send({ type: "replay", action: ui.replayClock && ui.replayClock.paused ? "play" : "pause" }));
  $("replay-restart").addEventListener("click", () => send({ type: "replay", action: "restart" }));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Pause") { event.preventDefault(); estop(); return; }
    if (event.key === "Escape") { disarm(); return; }
    if (typing(event)) return;
    if (event.key === "Enter" && !$("start-card").hidden) { event.preventDefault(); startEpisode(); return; }
    if (event.key.toLowerCase() === "t" && !event.repeat && !(ui.armed === "drive" && event.key.toLowerCase() in DRIVE_KEYS)) {
      event.preventDefault();
      toggleTelemetry();
      return;
    }
    const key = event.key.toLowerCase();
    if (ui.armed === "drive" && key in DRIVE_KEYS) { event.preventDefault(); ui.drive.keys.add(key); return; }
    if (event.key === "Enter") { event.preventDefault(); runArmed(); return; }
    const tile = document.querySelector(`.tile[data-key="${event.key}"]`);
    if (tile && !event.repeat) { event.preventDefault(); tile.click(); }
  });
  document.addEventListener("keyup", (event) => {
    ui.drive.keys.delete(event.key.toLowerCase());
    if (!ui.drive.keys.size) ui.drive.blocked = false;
  });
  window.addEventListener("blur", () => { ui.drive.keys.clear(); ui.drive.pad = [0, 0, 0]; });
  setInterval(driveTick, 200);
  setInterval(refreshCards, 500); // Elapsed times and distance to the goal.
  setupMainView();
  $("map-fit").addEventListener("click", fitMap);
  $("map-follow").addEventListener("click", () => {
    ui.mapView.follow = !ui.mapView.follow;
    if (ui.mapView.follow && ui.mapView.zoom < 2) ui.mapView.zoom = 2.5;
    markTabs();
  });
  for (const button of document.querySelectorAll("#mode-switch button")) {
    button.addEventListener("click", () => { ui.view.mode = button.dataset.mode; markTabs(); sendView(); });
  }
  for (const button of document.querySelectorAll("#log-filter button")) {
    button.addEventListener("click", () => {
      ui.logFilter = button.dataset.filter;
      for (const other of document.querySelectorAll("#log-filter button")) other.classList.toggle("on", other === button);
      for (const item of $("log").children) item.hidden = !visibleInLog(item);
    });
  }
  for (const channel of ["base", "arm"]) {
    const cancel = $(`card-${channel}`).querySelector(".cancel");
    cancel.addEventListener("click", () => send({ type: "cancel", id: cancel.dataset.id }));
  }
  const timeline = $("timeline"), tip = $("timeline-tip");
  timeline.addEventListener("mousemove", (event) => {
    const hit = ui.timelineHits.find(([x0, y0, x1, y1]) => event.offsetX >= x0 && event.offsetX <= x1 && event.offsetY >= y0 && event.offsetY <= y1);
    tip.hidden = !hit;
    if (!hit) return;
    const record = hit[4];
    tip.textContent = `${record.skill} #${record.id.replace(/^c/, "")} · ${resultText(record)} · ${fmt(record.started, 1)}–${fmt(record.ended == null ? simNow() : record.ended, 1)} s`;
    tip.style.left = `${Math.max(0, Math.min(event.offsetX + 12, timeline.clientWidth - 360))}px`;
  });
  timeline.addEventListener("mouseleave", () => { tip.hidden = true; });
  timeline.addEventListener("click", (event) => {
    const hit = ui.timelineHits.find(([x0, y0, x1, y1]) => event.offsetX >= x0 && event.offsetX <= x1 && event.offsetY >= y0 && event.offsetY <= y1);
    if (hit) inspectCommand(hit[4].id);
  });
  $("log").addEventListener("click", (event) => {
    const item = event.target.closest("li[data-command]");
    if (item) inspectCommand(item.dataset.command);
  });
  $("telemetry-toggle").addEventListener("click", () => toggleTelemetry());
  $("telemetry-close").addEventListener("click", () => toggleTelemetry(false));
  for (const button of document.querySelectorAll("#telemetry-tabs button")) {
    button.addEventListener("click", () => { ui.telemetry.tab = button.dataset.tab; renderTelemetry(); });
  }
  setInterval(() => { // Link throughput over the last second.
    ui.telemetry.rate = { bytes: ui.telemetry.bytes, frames: ui.telemetry.frames };
    ui.telemetry.bytes = ui.telemetry.frames = 0;
  }, 1000);
  $("pip").addEventListener("click", () => selectView(ui.view.main === "map" ? "front" : "map"));
  renderSkillPanel();
  renderLegend();
  connect();
  requestAnimationFrame(render);
}

setup();
