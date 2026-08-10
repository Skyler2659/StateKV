# StateKV control architecture v3 — visual contract

- Artifact: blank-slate architecture redesign; it does not derive layout from `statekv-architecture-v2.svg`.
- Target format: README full-width figure, 16:10 canvas, editable SVG plus PNG preview.
- Core claim: StateKV now spans an expensive state-conditioned physical oracle and a screen-positive cheap direct controller. Both act on the evolving compressed state; the cheap B3 route removes candidate model rollouts by producing token utilities and state-dependent layer budgets under a fixed global allocation.
- Reviewer question: What is evaluated by the physical oracle, what runs in the cheap controller, how does the chosen cache action change future state, and which parts have current evidence?
- Evidence layer: method overview plus a compact evidence/status strip.
- Source data: `README.md`; `configs/ccfa.yaml`; P26, P31, and P32 summary artifacts; `statekv/budget_dynamics.py`; `statekv/cheap_policy.py`.
- Statistics: P31 mean trajectory KL 0.05057; P31 preserves all five tested NIAH needles; P32 B3 mean trajectory KL 0.114995; P32 uses zero candidate model rollouts and preserves the fixed global core budget. These values are shown only in the evidence strip.
- Figure prototype: nested two-lane closed-loop system diagram with line-art module icons, followed by a four-stage evidence strip.
- Reader scan path: observed state on the left → oracle or cheap-controller lane → layer-wise cache action on the right → next token → feedback into the next observed state.
- Panel map:
  - Anchor: shared state/action loop.
  - Upper lane: physical oracle—candidate panel, physical rollouts, output risk, selection and stale-action regret.
  - Lower lane: B3 direct controller—online signals, token utility, layer difficulty, fixed-total budget allocation, per-layer top-k action.
  - Bottom strip: P26 mechanics, P31 oracle free generation, P32 cheap-controller screen, next mechanism/deployment gate.
- Typed connections: solid navy arrows are execution flow; dashed teal is evaluation reference rather than training; dashed orange is the next evidence gate; the long bottom arrow is state feedback.
- Training/inference boundary: the oracle is an expensive evaluator and reference controller. The B3 lane is training-free and does not learn from the oracle inside the depicted execution graph.
- Novel contribution emphasized: history-conditioned physical retained-set risk and the move from candidate rollouts to a direct layer-adaptive action.
- Required equations: `R_s(C)`, stale-action regret `G_s(C_old)`, token utility `q_{t,i}`, and the fixed-total layer-budget constraint.
- Exact label inventory: StateKV; Observed state; Active KV cache; Compression history; Physical oracle; Candidate action panel; Physical rollouts; Full-vocabulary risk; Select and refresh; Stale-action regret; B3 direct controller; Online signals; Token utility; Layer difficulty; Fixed-total allocator; Per-layer top-k; Layer-wise cache action; Next token; Cold KV backing; Evaluation reference; Mechanical loop; Oracle free generation; Cheap-controller screen; Mechanism and deployment gate.
- Caption role: explain that the oracle establishes the target closed-loop behavior while B3 is the current cheap-controller candidate; identify current evidence and the two next gates without turning the figure into a limitation list.
- Placement: proposed replacement for the current README architecture after user approval; no README edit in this iteration.
- Output formats: canonical editable SVG and high-resolution PNG preview.
- Traceability: all module names and numeric labels map to the listed repository artifacts; no external image generation and no invented result values.

## QA ledger

| Issue | Artifact | Severity | Fix | Status |
| --- | --- | --- | --- | --- |
| Rendered-layout inspection | v3 SVG/PNG | Medium | Shortened state/rollout labels, wrapped the action title, and separated the budget equation from its icon | Resolved |
| Dense label risk at README width | v3 SVG/PNG | Medium | Kept module labels short and moved result details into the bottom evidence strip | Resolved |
| Oracle-to-controller arrow could imply training | v3 SVG/PNG | High | Used a non-directional dashed divider labeled “evaluation reference · not an inference input” | Resolved |
| Cold backing path could imply input to decoding | v3 SVG/PNG | High | Routed the dashed recovery edge into the adaptive KV stack rather than the next-token decoder | Resolved |
| Color-only semantic separation | v3 PNG | Medium | Verified grayscale rendering; lane labels, borders, solid/dashed strokes, and module order preserve meaning without hue | Resolved |
| B3 lane tag overlapped the lane title | v3 SVG/PNG | High | Moved the "training-free candidate · P32" tag right of the measured title extent (x 238→272) | Resolved |
| Oracle lane tag text overflowed its pill | v3 SVG/PNG | Medium | Widened the "reference controller · P25–P31" pill from 170 to 236 units | Resolved |
| Evaluation-reference divider label overflowed its pill | v3 SVG/PNG | Medium | Widened the pill from 296 to 344 units and centered it on the divider line | Resolved |
| "one state · two control routes" overflowed its pill | v3 SVG/PNG | Low | Widened the pill from 230 to 250 units and centered the label | Resolved |
| Adaptive KV stack caption overlapped the L35 bar and ran past the panel | v3 SVG/PNG | High | Raised the panel to 216 units, moved the caption below the last bar row, and shortened the label to "fixed global budget · varying layers" | Resolved |
| Cold KV backing title and both description lines overflowed the dashed panel | v3 SVG/PNG | High | Restructured the panel: icon plus title on one row, two full-width tiny lines below, panel moved up to keep outer padding | Resolved |
| Vertical "recover" label overlapped the decode box | v3 SVG/PNG | Medium | Replaced the rotated label with a horizontal one in the gap between the decode box and the cold-backing panel | Resolved |
| "Layer-wise cache action" title wrapped 16/6 characters | v3 SVG/PNG | Low | Rebalanced the break to "Layer-wise" / "cache action" | Resolved |
| "exact KL / local risk" sat flush against the module edge | v3 SVG/PNG | Low | Shifted the pair left and changed the separator to a middle dot | Resolved |
| P32 card body line exceeded the card width | v3 SVG/PNG | Medium | Shortened to "fixed global allocation; best task-score point estimate" | Resolved |
| Layer-difficulty caption touched the L27 bar | v3 SVG/PNG | Low | Moved the bar rows up 7 units and the caption down 5 units | Resolved |
| SVG `<title>` disagreed with the hero heading | v3 SVG | Low | Aligned the accessible title with the hero text | Resolved |
| Headless-Chrome screenshot clipped the bottom ~50 units (window chrome/scrollbar) | v3 PNG | Medium | Rendered through a zero-margin HTML wrapper with `overflow:hidden` and an oversized window, then cropped to 3072×1920 | Resolved |
