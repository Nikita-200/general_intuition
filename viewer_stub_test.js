// viewer_stub_test.js — drives the REAL embedded JS simulation logic from
// a generated viewer_N.html, headless (no browser, no THREE.js needed),
// to confirm the task genuinely completes there too — not just that the
// Python reference simulation completes it.
//
// This exists because the JS engine embedded in every viewer_N.html
// (buildGrid/astar/planPathTo/controllerStep/resolveCollision/TaskRunner
// in viewer3d.py's JS_ENGINE block) is a from-scratch reimplementation of
// nav_agent_3d.py/tasks3d.py's logic, not a shared library — a real place
// for the two to drift apart silently. None of those functions reference
// THREE.js or the DOM (only the surrounding rendering code in the HTML
// template does), so this extracts JUST that block plus the embedded
// SCENE_DATA, evals them in plain Node, and re-drives a simulation loop
// that mirrors harness3d.py's own Python reference loop — checking real
// subgoal completion, the same ground truth harness3d.py itself prints.
//
// Usage: node viewer_stub_test.js out3d/viewer_0.html

const fs = require("fs");
const vm = require("vm");

const filePath = process.argv[2];
if (!filePath) {
  console.error("usage: node viewer_stub_test.js <path/to/viewer_N.html>");
  process.exit(1);
}
const html = fs.readFileSync(filePath, "utf8");

// Plain indexOf/slice, not a regex, on purpose: SCENE_DATA can be a large
// single-line JSON blob (embedded mesh data etc.), and a non-greedy regex
// scanning character-by-character for its closing "};\n" over a
// multi-megabyte line is needlessly slow (and, empirically, unreliable —
// silently failed to match at all against a real generated file).
const sceneMarker = "const SCENE_DATA = ";
const sceneStart = html.indexOf(sceneMarker);
if (sceneStart === -1) {
  console.error(`could not find "${sceneMarker}" in ${filePath}`);
  process.exit(1);
}
const sceneJsonStart = sceneStart + sceneMarker.length;
const sceneLineEnd = html.indexOf("\n", sceneJsonStart);
const sceneJsonText = html.slice(sceneJsonStart, sceneLineEnd).replace(/;\s*$/, "");
const SCENE_DATA = JSON.parse(sceneJsonText);

// JS_ENGINE sits between that SCENE_DATA line and the THREE.js setup that
// follows it (see viewer3d.py's HTML_TEMPLATE: "const SCENE_DATA = ...;\n"
// + JS_ENGINE + "\n\nconst scene3 = new THREE.Scene();"). Slicing between
// those two exact markers pulls out only the pure-logic block.
const engineStart = sceneLineEnd + 1;
const engineEndMarker = "const scene3 = new THREE.Scene();";
const engineEnd = html.indexOf(engineEndMarker, engineStart);
if (engineEnd === -1) {
  console.error("could not find the end of JS_ENGINE (THREE.Scene() marker) — " +
                "viewer3d.py's HTML_TEMPLATE structure may have changed; update this script's markers to match.");
  process.exit(1);
}
const jsEngineSrc = html.slice(engineStart, engineEnd);

const sandbox = { console };
vm.createContext(sandbox);
try {
  vm.runInContext(jsEngineSrc, sandbox, { filename: "JS_ENGINE (extracted)" });
  // `class`/`let`/`const` top-level declarations (TaskRunner is a class)
  // do NOT become properties of the context object the way top-level
  // `function` declarations do — that's true even for a vm context, not
  // just a browser's `window`. Running one more statement IN THE SAME
  // CONTEXT to alias them onto a `var` (which does attach) is the
  // straightforward way to actually get them out.
  vm.runInContext(
    "var __exports = { planPathTo, controllerStep, standoffDistance, physicalClearance, TaskRunner };",
    sandbox, { filename: "JS_ENGINE (export shim)" });
  Object.assign(sandbox, sandbox.__exports);
} catch (e) {
  console.error("JS_ENGINE block failed to even parse/run standalone:", e);
  process.exit(1);
}
const { planPathTo, controllerStep, standoffDistance, physicalClearance, TaskRunner } = sandbox;
for (const [name, fn] of Object.entries({ planPathTo, controllerStep, standoffDistance, physicalClearance, TaskRunner })) {
  if (typeof fn === "undefined") {
    console.error(`JS_ENGINE ran but didn't define ${name} — extraction markers may be off.`);
    process.exit(1);
  }
}

// ---- replicate harness3d.py's run_episode loop, against the JS engine ----
const state = {
  width: SCENE_DATA.width, depth: SCENE_DATA.depth, wallMargin: SCENE_DATA.wallMargin,
  assets: {}, markers: SCENE_DATA.markers, carried: null,
};
for (const name in SCENE_DATA.assets) {
  const a = SCENE_DATA.assets[name];
  state.assets[name] = { x: a.x, y: a.y, z: a.z, yaw: a.yaw, half_xz: a.half_xz, half_h: a.half_h, tags: a.tags.slice() };
}
const agentInit = SCENE_DATA.assets[SCENE_DATA.agentName];
const agent = { x: agentInit.x, y: agentInit.y, z: agentInit.z, yaw: agentInit.yaw, half_xz: agentInit.half_xz, half_h: agentInit.half_h };

const subgoals = SCENE_DATA.subgoals.map((s) => ({ ...s }));
const runner = new TaskRunner(subgoals);
let waypoints = null, wpIdx = 0, lastIdx = -1, arrivalThreshold = 30;
let stuckHistory = [];

function replan() {
  const sg = runner.current;
  if (!sg) { waypoints = null; return; }
  const target = runner.targetPos(state);
  const exclude = new Set([SCENE_DATA.agentName, sg.target]);
  if (sg.container) exclude.add(sg.container);
  waypoints = planPathTo(state, [agent.x, agent.z], target, exclude, agent.half_xz);
  wpIdx = 0; lastIdx = runner.idx;
  arrivalThreshold = waypoints
    ? Math.max(30, standoffDistance(waypoints, target) + 24, physicalClearance(state, agent, sg.target, 20))
    : 30;
  stuckHistory = [];
}

function isStuck() {
  stuckHistory.push([agent.x, agent.z]);
  if (stuckHistory.length > 45) stuckHistory.shift();
  if (stuckHistory.length < 45) return false;
  const [x0, z0] = stuckHistory[0], [x1, z1] = stuckHistory[stuckHistory.length - 1];
  const stuck = Math.hypot(x1 - x0, z1 - z0) < 6;
  if (stuck) stuckHistory = [];
  return stuck;
}

// Matches viewer3d.py's HTML_TEMPLATE updateCarried() exactly (forward
// offset 26, chest height 110) — kept in sync by hand since it lives
// outside the extracted JS_ENGINE block (it's rendering-adjacent glue in
// the HTML template itself, not pure simulation logic).
function updateCarried() {
  if (!state.carried) return;
  const item = state.assets[state.carried];
  const fwd = 26;
  item.x = agent.x + fwd * Math.sin(agent.yaw);
  item.z = agent.z + fwd * Math.cos(agent.yaw);
  item.y = 110;
}

const MAX_TICKS = 1400;
let t = 0, noPath = false;
for (; t < MAX_TICKS; t++) {
  if (runner.isComplete) break;
  if (runner.idx !== lastIdx) replan();
  if (!waypoints) { noPath = true; break; }
  const excludeSet = new Set([SCENE_DATA.agentName]);
  if (runner.current) excludeSet.add(runner.current.target);
  const r = controllerStep(state, agent, waypoints, wpIdx, excludeSet, 140, 1 / 60);
  wpIdx = r.wpIdx;
  if (isStuck()) replan();
  updateCarried();
  runner.tryAdvance(state, agent, arrivalThreshold || 30);
}

const result = {
  file: filePath, ticks: t, complete: runner.isComplete, noPath,
  subgoals: subgoals.map((s) => `${s.done ? "[x]" : "[ ]"} ${s.kind} ${s.target}${s.container ? " -> " + s.container : ""}`),
};
console.log(JSON.stringify(result, null, 2));

if (!runner.isComplete) {
  console.error(`FAIL: task did not complete within ${MAX_TICKS} ticks (noPath=${noPath})`);
  process.exit(1);
}
console.log(`PASS: the embedded JS engine genuinely completed the task in ${t} ticks.`);