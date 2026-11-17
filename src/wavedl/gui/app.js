const ICONS = {
  check: '<path d="M20 6 9 17l-5-5"/>',
  file: '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="m10 13-2 2 2 2"/><path d="m14 17 2-2-2-2"/>',
  open: '<path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/>',
  folder: '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
  save: '<path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7"/><path d="M7 3v4a1 1 0 0 0 1 1h7"/>',
  play: '<polygon points="6 3 20 12 6 21 6 3" fill="currentColor"/>',
  stop: '<rect x="5" y="5" width="14" height="14" rx="2" fill="currentColor"/>',
  reset: '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
  copy: '<rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
  terminal: '<polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/>',
  panel: '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M15 3v18"/><path d="m8 9 3 3-3 3"/>',
  left: '<path d="m12 19-7-7 7-7"/><path d="M19 12H5"/>',
  right: '<path d="m9 18 6-6-6-6"/>',
  up: '<path d="m18 15-6-6-6 6"/>',
  search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
  db: '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
  layers: '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
  wave: '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
  target: '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
  grid: '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/><path d="M15 3v18"/>',
  star: '<path fill="currentColor" stroke="none" d="M11.525 2.295a.53.53 0 0 1 .95 0l2.31 4.679a2.123 2.123 0 0 0 1.595 1.16l5.166.756a.53.53 0 0 1 .294.904l-3.736 3.638a2.123 2.123 0 0 0-.611 1.878l.882 5.14a.53.53 0 0 1-.771.56l-4.618-2.428a2.122 2.122 0 0 0-1.973 0L6.396 21.01a.53.53 0 0 1-.77-.56l.881-5.139a2.122 2.122 0 0 0-.611-1.879L2.16 9.795a.53.53 0 0 1 .294-.906l5.165-.755a2.122 2.122 0 0 0 1.597-1.16z"/>',
  zap: '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
  split: '<rect width="7" height="18" x="3" y="3" rx="1"/><rect width="7" height="7" x="14" y="3" rx="1"/><rect width="7" height="7" x="14" y="14" rx="1"/>',
  scale: '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
  chart: '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="m19 9-5 5-4-4-3 3"/>',
  package: '<path d="m7.5 4.27 9 5.15"/><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>',
  sparkle: '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>',
  alert: '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>',
  home: '<path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1Z"/>',
};

function icon(name, className = "") {
  return `<svg class="icon ${className}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ""}</svg>`;
}

const MODEL_CATALOG = [
  { name: "cnn", cat: "Classic CNN", dims: ["1D", "2D", "3D"], note: "Fast universal baseline" },
  { name: "resnet18", cat: "Classic CNN", dims: ["1D", "2D", "3D"], note: "Residual network (18 layers)" },
  { name: "resnet34", cat: "Classic CNN", dims: ["1D", "2D", "3D"], note: "Residual network (34 layers)" },
  { name: "resnet50", cat: "Classic CNN", dims: ["1D", "2D", "3D"], note: "Bottleneck ResNet (50 layers)" },
  { name: "resnet18_pretrained", cat: "Classic CNN", dims: ["2D"], pretrained: true, note: "ImageNet pretrained ResNet-18" },
  { name: "resnet50_pretrained", cat: "Classic CNN", dims: ["2D"], pretrained: true, note: "ImageNet pretrained ResNet-50" },
  { name: "densenet121", cat: "Classic CNN", dims: ["1D", "2D", "3D"], note: "Dense feature reuse (121L)" },
  { name: "densenet169", cat: "Classic CNN", dims: ["1D", "2D", "3D"], note: "Dense feature reuse (169L)" },
  { name: "densenet121_pretrained", cat: "Classic CNN", dims: ["2D"], pretrained: true, note: "ImageNet DenseNet-121" },
  { name: "mobilenet_v3_small", cat: "Efficient CNN", dims: ["2D"], pretrained: true, note: "Ultra-fast lightweight 2D" },
  { name: "mobilenet_v3_large", cat: "Efficient CNN", dims: ["2D"], pretrained: true, note: "MobileNetV3 high capacity" },
  { name: "efficientnet_b0", cat: "Efficient CNN", dims: ["2D"], pretrained: true, note: "Compound-scaled CNN (B0)" },
  { name: "efficientnet_b2", cat: "Efficient CNN", dims: ["2D"], pretrained: true, note: "Compound-scaled CNN (B2)" },
  { name: "efficientnet_b4", cat: "Efficient CNN", dims: ["2D"], pretrained: true, note: "Compound-scaled CNN (B4)" },
  { name: "efficientnet_b7", cat: "Efficient CNN", dims: ["2D"], pretrained: true, note: "Compound-scaled CNN (B7)" },
  { name: "efficientnet_v2_s", cat: "Efficient CNN", dims: ["2D"], pretrained: true, note: "Fast progressive training (S)" },
  { name: "efficientnet_v2_m", cat: "Efficient CNN", dims: ["2D"], pretrained: true, note: "EfficientNetV2 Medium" },
  { name: "efficientnet_v2_l", cat: "Efficient CNN", dims: ["2D"], pretrained: true, note: "EfficientNetV2 Large" },
  { name: "regnet_y_400mf", cat: "Efficient CNN", dims: ["2D"], pretrained: true, note: "RegNetY 400M FLOPs" },
  { name: "regnet_y_800mf", cat: "Efficient CNN", dims: ["2D"], pretrained: true, note: "RegNetY 800M FLOPs" },
  { name: "regnet_y_1_6gf", cat: "Efficient CNN", dims: ["2D"], pretrained: true, note: "RegNetY 1.6G FLOPs" },
  { name: "regnet_y_3_2gf", cat: "Efficient CNN", dims: ["2D"], pretrained: true, note: "RegNetY 3.2G FLOPs" },
  { name: "regnet_y_8gf", cat: "Efficient CNN", dims: ["2D"], pretrained: true, note: "RegNetY 8.0G FLOPs" },
  { name: "convnext_tiny", cat: "Modern CNN", dims: ["1D", "2D", "3D"], note: "7x7 depthwise modern CNN" },
  { name: "convnext_small", cat: "Modern CNN", dims: ["1D", "2D", "3D"], note: "ConvNeXt Small" },
  { name: "convnext_base", cat: "Modern CNN", dims: ["1D", "2D", "3D"], note: "ConvNeXt Base" },
  { name: "convnext_tiny_pretrained", cat: "Modern CNN", dims: ["2D"], pretrained: true, note: "ImageNet ConvNeXt-Tiny" },
  { name: "convnext_v2_tiny", cat: "Modern CNN", dims: ["1D", "2D", "3D"], note: "GRN-enhanced ConvNeXt V2" },
  { name: "convnext_v2_small", cat: "Modern CNN", dims: ["1D", "2D", "3D"], note: "ConvNeXt V2 Small" },
  { name: "convnext_v2_base", cat: "Modern CNN", dims: ["1D", "2D", "3D"], note: "ConvNeXt V2 Base" },
  { name: "convnext_v2_tiny_pretrained", cat: "Modern CNN", dims: ["2D"], pretrained: true, note: "ImageNet ConvNeXt-V2" },
  { name: "unireplknet_tiny", cat: "Modern CNN", dims: ["1D", "2D", "3D"], note: "Large-kernel re-param CNN" },
  { name: "unireplknet_small", cat: "Modern CNN", dims: ["1D", "2D", "3D"], note: "UniRepLKNet Small" },
  { name: "unireplknet_base", cat: "Modern CNN", dims: ["1D", "2D", "3D"], note: "UniRepLKNet Base" },
  { name: "vit_tiny", cat: "Transformer", dims: ["1D", "2D"], note: "Patch Vision Transformer" },
  { name: "vit_small", cat: "Transformer", dims: ["1D", "2D"], note: "ViT Small" },
  { name: "vit_base", cat: "Transformer", dims: ["1D", "2D"], note: "ViT Base" },
  { name: "swin_t", cat: "Transformer", dims: ["2D"], pretrained: true, note: "Shifted-window Transformer" },
  { name: "swin_s", cat: "Transformer", dims: ["2D"], pretrained: true, note: "Swin Small" },
  { name: "swin_b", cat: "Transformer", dims: ["2D"], pretrained: true, note: "Swin Base" },
  { name: "maxvit_tiny", cat: "Transformer", dims: ["2D"], pretrained: true, note: "Multi-axis attention (Tiny)" },
  { name: "maxvit_small", cat: "Transformer", dims: ["2D"], pretrained: true, note: "MaxViT Small" },
  { name: "maxvit_base", cat: "Transformer", dims: ["2D"], pretrained: true, note: "MaxViT Base" },
  { name: "fastvit_t8", cat: "Hybrid", dims: ["2D"], pretrained: true, note: "Reparam hybrid ViT (T8)" },
  { name: "fastvit_t12", cat: "Hybrid", dims: ["2D"], pretrained: true, note: "FastViT T12" },
  { name: "fastvit_s12", cat: "Hybrid", dims: ["2D"], pretrained: true, note: "FastViT S12" },
  { name: "fastvit_sa12", cat: "Hybrid", dims: ["2D"], pretrained: true, note: "FastViT SA12" },
  { name: "caformer_s18", cat: "Hybrid", dims: ["2D"], pretrained: true, note: "Conv + Attention MetaFormer" },
  { name: "caformer_s36", cat: "Hybrid", dims: ["2D"], pretrained: true, note: "CAFormer S36" },
  { name: "caformer_m36", cat: "Hybrid", dims: ["2D"], pretrained: true, note: "CAFormer M36" },
  { name: "poolformer_s12", cat: "Hybrid", dims: ["2D"], pretrained: true, note: "Pooling MetaFormer S12" },
  { name: "efficientvit_m1", cat: "Hybrid", dims: ["2D"], pretrained: true, note: "Cascaded group attention" },
  { name: "efficientvit_b1", cat: "Hybrid", dims: ["2D"], pretrained: true, note: "EfficientViT B1" },
  { name: "efficientvit_b2", cat: "Hybrid", dims: ["2D"], pretrained: true, note: "EfficientViT B2" },
  { name: "efficientvit_l2", cat: "Hybrid", dims: ["2D"], pretrained: true, note: "EfficientViT L2" },
  { name: "mamba_1d", cat: "SSM / Mamba", dims: ["1D"], note: "Selective State Space (1D)" },
  { name: "vim_tiny", cat: "SSM / Mamba", dims: ["2D"], note: "Bidirectional Vision Mamba" },
  { name: "vim_small", cat: "SSM / Mamba", dims: ["2D"], note: "Vision Mamba Small" },
  { name: "vim_base", cat: "SSM / Mamba", dims: ["2D"], note: "Vision Mamba Base" },
  { name: "s4d_small", cat: "SSM / Mamba", dims: ["1D"], note: "Diagonal State Space (1D)" },
  { name: "s4d", cat: "SSM / Mamba", dims: ["1D"], note: "S4D Standard (long 1D)" },
  { name: "s4d_large", cat: "SSM / Mamba", dims: ["1D"], note: "S4D Large" },
  { name: "tcn_small", cat: "Specialized", dims: ["1D"], note: "Dilated causal TCN (Small)" },
  { name: "tcn", cat: "Specialized", dims: ["1D"], note: "Temporal ConvNet for 1D waves" },
  { name: "tcn_large", cat: "Specialized", dims: ["1D"], note: "Temporal ConvNet (Large)" },
  { name: "wavenet_small", cat: "Specialized", dims: ["1D"], note: "Gated dilated WaveNet (Small)" },
  { name: "wavenet", cat: "Specialized", dims: ["1D"], note: "WaveNet for ultrasonic/audio" },
  { name: "wavenet_large", cat: "Specialized", dims: ["1D"], note: "WaveNet (Large)" },
  { name: "resnet3d_18", cat: "Specialized", dims: ["3D"], pretrained: true, note: "3D volumetric ResNet-18" },
  { name: "mc3_18", cat: "Specialized", dims: ["3D"], pretrained: true, note: "Mixed 3D/2D spatio-temporal" },
  { name: "unet_regression", cat: "Specialized", dims: ["1D", "2D", "3D"], note: "U-Net encoder-decoder → Vector" },
  { name: "unet_field", cat: "Specialized", dims: ["2D", "field"], note: "Dense 2D field map (N,H,W)" },
  { name: "inversionnet", cat: "Specialized", dims: ["2D", "field"], note: "OpenFWI 5×1000×70 → 70×70 velocity" },
  { name: "velocitygan", cat: "Specialized", dims: ["2D", "field"], note: "OpenFWI GAN; needs --adversarial wgan-gp" }
];

const DEFAULTS = {
  model: "cnn",
  pretrained: false,
  import_modules: [],
  batch_size: 128,
  lr: 0.001,
  epochs: 1000,
  patience: 20,
  weight_decay: 0.0001,
  grad_clip: 1.0,
  grad_accum_steps: 1,
  loss: "mse",
  selection_metric: "loss",
  huber_delta: 1.0,
  loss_weights: "",
  optimizer: "adamw",
  momentum: 0.9,
  nesterov: false,
  betas: "0.9,0.999",
  scheduler: "plateau",
  scheduler_factor: 0.5,
  scheduler_patience: 10,
  min_lr: 0.000001,
  warmup_epochs: 5,
  step_size: 30,
  milestones: "",
  cv: 0,
  cv_stratify: false,
  cv_bins: 10,
  data_path: "",
  output_dir: "./experiments",
  resume: "",
  save_every: 50,
  fresh: false,
  workers: -1,
  seed: 2025,
  deterministic: false,
  cache_validate: "sha256",
  single_channel: false,
  allow_pickle: false,
  precision: "bf16",
  compile: false,
  wandb: false,
  wandb_watch: false,
  project_name: "DL-Training",
  run_name: "",
  constraint: [],
  constraint_file: "",
  constraint_weight: [0.1],
  constraint_reduction: "mse"
};

let state = JSON.parse(JSON.stringify(DEFAULTS));
let extraConfigKeys = {};
let activeModelCat = "All";
let activeDimFilter = "all";
let dataFileCheck = { exists: false, is_file: false, size_mb: null };
let pollTimer = null;
let logOffset = 0;
let activeJobId = null;
let wasJobRunning = false;

function postHeaders() {
  const headers = { "Content-Type": "application/json" };
  const meta = document.querySelector('meta[name="wavedl-token"]');
  if (meta && meta.content) {
    headers["X-WaveDL-Token"] = meta.content;
  }
  return headers;
}

function updateThemeButton() {
  const theme = document.documentElement.dataset.theme || "auto";
  const labels = { auto: "Auto", light: "Light", dark: "Dark" };
  const next = { auto: "light", light: "dark", dark: "auto" }[theme];
  const icons = {
    auto: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
    light: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42"/>',
    dark: '<path d="M20.9 13A9 9 0 0 1 11 3.1 9 9 0 1 0 20.9 13Z"/>'
  };
  const button = document.getElementById("themeToggle");
  const label = `Theme: ${labels[theme]}. Switch to ${labels[next]}`;
  button.title = label;
  button.setAttribute("aria-label", label);
  button.innerHTML = `<svg aria-hidden="true" class="icon" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">${icons[theme]}</svg>`;
}

function cycleTheme() {
  const theme = document.documentElement.dataset.theme || "auto";
  const next = { auto: "light", light: "dark", dark: "auto" }[theme];
  if (next === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("wavedl-theme", next);
  } catch (_) {}
  updateThemeButton();
}

function init() {
  updateThemeButton();
  document.getElementById("stepperNav").addEventListener("keydown", e => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const buttons = [...document.querySelectorAll("#stepperNav .step-btn")];
    const index = buttons.indexOf(document.activeElement);
    if (index === -1) return;
    e.preventDefault();
    const next = Math.max(0, Math.min(buttons.length - 1, index + (e.key === "ArrowRight" ? 1 : -1)));
    goToStep(next + 1);
    buttons[next].focus();
  });
  const cats = ["All", ...new Set(MODEL_CATALOG.map(m => m.cat))];
  document.getElementById("modelCatPills").innerHTML = cats.map(c =>
    `<button type="button" class="pill ${c === activeModelCat ? "active" : ""}" onclick="setModelCat('${c}')">${c === "All" ? "All families" : c}</button>`
  ).join("");

  if (window.innerWidth <= 760) toggleConfigPane();
  goToStep(1);
  applyStateToDOM();
  loadConfigFromFile(true);
  checkTrainingStatus();
}

function goToStep(stepNum) {
  for (let i = 1; i <= 5; i++) {
    document.getElementById(`panel-step-${i}`).classList.toggle("active", i === stepNum);
  }
  document.querySelectorAll("#stepperNav .step-btn").forEach((b, idx) => {
    b.classList.toggle("active", idx + 1 === stepNum);
    if (idx + 1 === stepNum) b.setAttribute("aria-current", "step");
    else b.removeAttribute("aria-current");
  });
  document.querySelector(".wizard-pane").scrollTop = 0;
}

function setConsoleDrawer(open) {
  document.getElementById("consoleDrawer").classList.toggle("open", open);
  document.getElementById("consoleToggle").setAttribute("aria-expanded", String(open));
}

function toggleConsoleDrawer() {
  setConsoleDrawer(!document.getElementById("consoleDrawer").classList.contains("open"));
}

function toggleConfigPane() {
  const hidden = document.querySelector(".right-pane").classList.toggle("hidden");
  document.getElementById("showConfigBtn").hidden = !hidden;
}

function useSuggestedCheckpoint() {
  document.getElementById("t_checkpoint").value = document.getElementById("checkpointSuggestion").textContent;
}

function applyPreset(name) {
  if (name === "baseline") {
    Object.assign(state, {
      model: "cnn", pretrained: false, batch_size: 128, lr: 0.001,
      optimizer: "adamw", scheduler: "plateau", loss: "mse", precision: "bf16", compile: false, cv: 0
    });
    showToast("Template applied: Standard Baseline (CNN)");
  } else if (name === "convnext") {
    Object.assign(state, {
      model: "convnext_v2_tiny", pretrained: false, batch_size: 64, lr: 0.0005,
      optimizer: "adamw", scheduler: "cosine", loss: "mse", precision: "bf16", compile: true, cv: 0
    });
    showToast("Template applied: ConvNeXt V2 + Cosine");
  } else if (name === "waveform1d") {
    Object.assign(state, {
      model: "tcn", pretrained: false, batch_size: 128, lr: 0.003,
      optimizer: "adamw", scheduler: "onecycle", loss: "mse", precision: "bf16", cv: 0
    });
    showToast("Template applied: 1D Waveform (TCN)");
  } else if (name === "field2d") {
    Object.assign(state, {
      model: "unet_field", pretrained: false, batch_size: 32, lr: 0.001,
      optimizer: "adamw", scheduler: "cosine", loss: "mse", precision: "bf16", cv: 0
    });
    showToast("Template applied: 2D Property Map (U-Net Field)");
  }
  applyStateToDOM();
}

function setDimFilter(dim) {
  activeDimFilter = dim;
  document.querySelectorAll("#dimFilterPills .pill").forEach(p => {
    p.classList.toggle("active", p.getAttribute("data-dim") === dim);
  });
  renderModelGrid();
}

function setModelCat(cat) {
  activeModelCat = cat;
  document.querySelectorAll("#modelCatPills .pill").forEach(p => {
    p.classList.toggle("active", p.textContent === (cat === "All" ? "All families" : cat));
  });
  renderModelGrid();
}

function renderModelGrid() {
  const q = (document.getElementById("modelSearch").value || "").toLowerCase().trim();
  const grid = document.getElementById("modelGrid");
  const filtered = MODEL_CATALOG.filter(m => {
    const matchCat = activeModelCat === "All" || m.cat === activeModelCat;
    const matchDim = activeDimFilter === "all" || m.dims.includes(activeDimFilter);
    const matchQ = !q || m.name.toLowerCase().includes(q) || m.cat.toLowerCase().includes(q) || m.note.toLowerCase().includes(q);
    return matchCat && matchDim && matchQ;
  });
  grid.innerHTML = filtered.map(m => {
    const selected = state.model === m.name;
    return `<button type="button" class="model-card ${selected ? "active" : ""}" onclick="chooseModel('${m.name}')" title="${m.name}" aria-pressed="${selected}">
      <span class="model-card-top"><span class="model-id">${m.name}</span>${m.pretrained ? icon("star", "model-star") : ""}</span>
      ${selected ? `<span class="model-check">${icon("check")}</span>` : ""}
      <span class="model-meta">${m.note}</span>
      <span class="model-card-bottom">${m.dims.map(d => `<span class="dim-badge">${d}</span>`).join("")}<span class="model-family">${m.cat}</span></span>
    </button>`;
  }).join("") || '<div class="model-empty">No models match. Try another search or choose All.</div>';
}

function chooseModel(name) {
  state.model = name;
  document.getElementById("f_model").value = name;
  if (name.endsWith("_pretrained")) {
    state.pretrained = true;
    document.getElementById("f_pretrained").checked = true;
  }
  renderModelGrid();
  syncFromForm();
}

function onModelInput() {
  state.model = document.getElementById("f_model").value.trim() || "cnn";
  renderModelGrid();
  syncFromForm();
}

let pathCheckTimer = null;
function onDataPathChange() {
  clearTimeout(pathCheckTimer);
  dataFileCheck = { exists: false, is_file: null, size_mb: null };
  syncFromForm();
  const p = (document.getElementById("f_data_path").value || "").trim();
  if (!p) {
    dataFileCheck = { exists: false, is_file: false, size_mb: null };
    updateSummaryAndBadges();
    return;
  }
  pathCheckTimer = setTimeout(() => verifyDataPath(p), 220);
}

async function verifyDataPath(pathStr) {
  try {
    const res = await fetch(`/api/check_path?path=${encodeURIComponent(pathStr)}`);
    if (state.data_path !== pathStr) return;
    if (!res.ok) {
      dataFileCheck = { exists: false, is_file: false, size_mb: null };
    } else {
      dataFileCheck = await res.json();
    }
    updateSummaryAndBadges();
  } catch (_) {}
}

function selectPill(field, val) {
  const prev = state[field];
  state[field] = val;
  if (field === "scheduler") {
    const sfEl = document.getElementById("f_scheduler_factor");
    if (val === "exponential" && state.scheduler_factor === 0.5) {
      state.scheduler_factor = 0.95;
      if (sfEl) sfEl.value = 0.95;
    } else if (prev === "exponential" && val !== "exponential" && state.scheduler_factor === 0.95) {
      state.scheduler_factor = 0.5;
      if (sfEl) sfEl.value = 0.5;
    }
  }
  const container = document.getElementById(`pills_${field}`);
  if (container) {
    container.querySelectorAll(".pill").forEach(p => {
      p.classList.toggle("active", p.getAttribute("data-val") === val);
    });
  }
  updateConditionalFields();
  syncFromForm();
}

function setVal(id, val) {
  const el = document.getElementById(id);
  if (el) {
    el.value = val;
    syncFromForm();
  }
}





function onToggleCV(enabled) {
  state.cv = enabled ? Math.max(2, state.cv || 5) : 0;
  document.getElementById("f_cv").value = state.cv;
  updateConditionalFields();
  syncFromForm();
}

function onToggleConstraints(enabled) {
  document.getElementById("constraintsDetailsBox").style.display = enabled ? "" : "none";
  if (!enabled) {
    document.getElementById("f_constraint").value = "";
    document.getElementById("f_constraint_file").value = "";
  }
  syncFromForm();
}

function updateConditionalFields() {
  const isHuber = state.loss === "huber";
  const isWeighted = state.loss === "weighted_mse";
  document.getElementById("lossContextBox").style.display = (isHuber || isWeighted) ? "" : "none";
  document.getElementById("wrap_huber_delta").style.display = isHuber ? "" : "none";
  document.getElementById("wrap_loss_weights").style.display = isWeighted ? "" : "none";

  const isSgd = state.optimizer === "sgd";
  const isRms = state.optimizer === "rmsprop";
  const isAdam = ["adamw", "adam", "nadam", "radam"].includes(state.optimizer);
  document.getElementById("wrap_momentum").style.display = (isSgd || isRms) ? "" : "none";
  document.getElementById("wrap_nesterov").style.display = isSgd ? "" : "none";
  document.getElementById("wrap_betas").style.display = isAdam ? "" : "none";

  const sc = state.scheduler;
  document.getElementById("wrap_scheduler_patience").style.display = sc === "plateau" ? "" : "none";
  document.getElementById("wrap_scheduler_factor").style.display = ["plateau", "step", "multistep", "exponential"].includes(sc) ? "" : "none";
  document.getElementById("wrap_warmup_epochs").style.display = sc === "linear_warmup" ? "" : "none";
  document.getElementById("wrap_step_size").style.display = sc === "step" ? "" : "none";
  document.getElementById("wrap_milestones").style.display = sc === "multistep" ? "" : "none";

  const cvOn = state.cv >= 2;
  document.getElementById("toggle_cv_enabled").checked = cvOn;
  document.getElementById("cvDetailsBox").style.display = cvOn ? "" : "none";

  const hasConstraints = (Array.isArray(state.constraint) && state.constraint.length > 0) || Boolean(state.constraint_file);
  const cToggle = document.getElementById("toggle_constraints_enabled");
  if (hasConstraints) cToggle.checked = true;
  document.getElementById("constraintsDetailsBox").style.display = cToggle.checked ? "" : "none";

  document.getElementById("wandbDetailsBox").style.display = state.wandb ? "" : "none";
}

function updateSummaryAndBadges() {
  const dataStatus = document.getElementById("dataFileStatus");
  const missing = !state.data_path || dataFileCheck.is_file === false;
  document.querySelector("#stepperNav .step-btn").classList.toggle("warn", missing);
  dataStatus.classList.toggle("warn", missing);
  if (!state.data_path) {
    dataStatus.textContent = "";
  } else if (dataFileCheck.is_file) {
    const size = dataFileCheck.size_mb != null ? `${dataFileCheck.size_mb} MB` : "File found";
    dataStatus.innerHTML = icon("check");
    dataStatus.append(document.createTextNode(size));
  } else {
    dataStatus.textContent = dataFileCheck.is_file === null ? "Checking…" : "File missing";
  }
  const wLabel = state.workers === -1 ? "auto" : state.workers;
  document.getElementById("advDataSummary").textContent =
    `seed ${state.seed} · workers ${wLabel} · save every ${state.save_every}`;
  document.getElementById("advTrainingSummary").textContent =
    `decay ${state.weight_decay} · clip ${state.grad_clip} · accum ${state.grad_accum_steps}`;
  document.getElementById("checkpointSuggestion").textContent =
    `${(state.output_dir || ".").replace(/[\\/]+$/, "")}/best_checkpoint`;
  document.querySelectorAll(".number-presets .pill").forEach(p => {
    const match = p.getAttribute("onclick").match(/setVal\('([^']+)',\s*([\d.eE+-]+)\)/);
    if (match) p.classList.toggle("active", Number(document.getElementById(match[1]).value) === Number(match[2]));
  });
}

function applyStateToDOM() {
  const numOrText = [
    "model", "batch_size", "lr", "epochs", "patience", "weight_decay",
    "grad_clip", "grad_accum_steps", "huber_delta", "loss_weights",
    "momentum", "betas", "scheduler_factor", "scheduler_patience",
    "min_lr", "warmup_epochs", "step_size", "milestones", "cv", "cv_bins",
    "data_path", "output_dir", "resume", "save_every", "workers", "seed",
    "project_name", "run_name", "constraint_file"
  ];
  numOrText.forEach(k => {
    const el = document.getElementById(`f_${k}`);
    if (el) el.value = state[k] !== undefined && state[k] !== null ? state[k] : "";
  });

  document.getElementById("f_import_modules").value = Array.isArray(state.import_modules)
    ? state.import_modules.join(", ")
    : (state.import_modules || "");

  document.getElementById("f_constraint").value = Array.isArray(state.constraint)
    ? state.constraint.join("; ")
    : (state.constraint || "");

  document.getElementById("f_constraint_weight").value = Array.isArray(state.constraint_weight)
    ? state.constraint_weight.join(", ")
    : (state.constraint_weight !== undefined ? state.constraint_weight : "0.1");

  const bools = [
    "pretrained", "nesterov", "cv_stratify", "fresh", "deterministic",
    "single_channel", "allow_pickle", "compile", "wandb", "wandb_watch"
  ];
  bools.forEach(k => {
    const el = document.getElementById(`f_${k}`);
    if (el) el.checked = Boolean(state[k]);
  });

  ["loss", "selection_metric", "optimizer", "scheduler", "cache_validate", "precision", "constraint_reduction"].forEach(k => {
    const container = document.getElementById(`pills_${k}`);
    if (container) {
      container.querySelectorAll(".pill").forEach(p => {
        p.classList.toggle("active", p.getAttribute("data-val") === String(state[k]));
      });
    }
  });

  dataFileCheck = { exists: false, is_file: state.data_path ? null : false, size_mb: null };
  updateConditionalFields();
  updateSummaryAndBadges();
  renderModelGrid();
  renderYamlPreview();
  if (state.data_path) verifyDataPath(state.data_path);
  else updateSummaryAndBadges();
}

function readDOMToState() {
  const getNum = (id, fallback) => {
    const v = parseFloat(document.getElementById(id).value);
    return Number.isFinite(v) ? v : fallback;
  };
  const getInt = (id, fallback) => {
    const v = parseInt(document.getElementById(id).value, 10);
    return Number.isFinite(v) ? v : fallback;
  };
  const getStr = (id) => (document.getElementById(id).value || "").trim();
  const getBool = (id) => Boolean(document.getElementById(id).checked);

  state.model = getStr("f_model") || "cnn";
  state.pretrained = getBool("f_pretrained");
  state.import_modules = getStr("f_import_modules")
    ? getStr("f_import_modules").split(",").map(s => s.trim()).filter(Boolean)
    : [];

  state.batch_size = getInt("f_batch_size", 128);
  state.lr = getNum("f_lr", 0.001);
  state.epochs = getInt("f_epochs", 1000);
  state.patience = getInt("f_patience", 20);
  state.weight_decay = getNum("f_weight_decay", 0.0001);
  state.grad_clip = getNum("f_grad_clip", 1.0);
  state.grad_accum_steps = getInt("f_grad_accum_steps", 1);

  state.huber_delta = getNum("f_huber_delta", 1.0);
  state.loss_weights = getStr("f_loss_weights");

  state.momentum = getNum("f_momentum", 0.9);
  state.nesterov = getBool("f_nesterov");
  state.betas = getStr("f_betas") || "0.9,0.999";

  state.scheduler_factor = getNum("f_scheduler_factor", 0.5);
  state.scheduler_patience = getInt("f_scheduler_patience", 10);
  state.min_lr = getNum("f_min_lr", 1e-6);
  state.warmup_epochs = getInt("f_warmup_epochs", 5);
  state.step_size = getInt("f_step_size", 30);
  state.milestones = getStr("f_milestones");

  state.cv = document.getElementById("toggle_cv_enabled").checked
    ? Math.max(2, getInt("f_cv", 5))
    : 0;
  state.cv_stratify = getBool("f_cv_stratify");
  state.cv_bins = getInt("f_cv_bins", 10);

  state.data_path = getStr("f_data_path");
  state.output_dir = getStr("f_output_dir");
  state.resume = getStr("f_resume");
  state.save_every = getInt("f_save_every", 50);
  state.fresh = getBool("f_fresh");
  state.workers = getInt("f_workers", -1);
  state.seed = getInt("f_seed", 2025);
  state.deterministic = getBool("f_deterministic");
  state.single_channel = getBool("f_single_channel");
  state.allow_pickle = getBool("f_allow_pickle");

  state.compile = getBool("f_compile");
  state.wandb = getBool("f_wandb");
  state.wandb_watch = getBool("f_wandb_watch");
  state.project_name = getStr("f_project_name");
  state.run_name = getStr("f_run_name");

  state.constraint = getStr("f_constraint")
    ? getStr("f_constraint").split(/[;\n]+/).map(s => s.trim()).filter(Boolean)
    : [];
  state.constraint_file = getStr("f_constraint_file");
  const cwRaw = getStr("f_constraint_weight");
  state.constraint_weight = cwRaw
    ? cwRaw.split(",").map(s => parseFloat(s.trim())).filter(Number.isFinite)
    : [0.1];
}

function buildEffectiveConfig() {
  const cfg = {
    model: state.model,
    pretrained: state.pretrained,
    batch_size: state.batch_size,
    lr: state.lr,
    epochs: state.epochs,
    patience: state.patience,
    weight_decay: state.weight_decay,
    grad_clip: state.grad_clip,
    grad_accum_steps: state.grad_accum_steps,
    loss: state.loss,
    optimizer: state.optimizer,
    scheduler: state.scheduler,
    cv: state.cv,
    cv_stratify: state.cv_stratify,
    cv_bins: state.cv_bins,
    workers: state.workers,
    seed: state.seed,
    deterministic: state.deterministic,
    cache_validate: state.cache_validate,
    precision: state.precision,
    compile: state.compile
  };

  if (state.import_modules && state.import_modules.length > 0) cfg.import_modules = state.import_modules;
  if (state.selection_metric && state.selection_metric !== "loss") cfg.selection_metric = state.selection_metric;
  if (state.loss === "huber") cfg.huber_delta = state.huber_delta;
  if (state.loss === "weighted_mse" && state.loss_weights) cfg.loss_weights = state.loss_weights;

  if (state.optimizer === "sgd" || state.optimizer === "rmsprop") cfg.momentum = state.momentum;
  if (state.optimizer === "sgd" && state.nesterov) cfg.nesterov = true;
  if (["adamw", "adam", "nadam", "radam"].includes(state.optimizer) && state.betas !== "0.9,0.999") {
    cfg.betas = state.betas;
  }

  if (["plateau", "step", "multistep"].includes(state.scheduler)) {
    cfg.scheduler_factor = state.scheduler_factor;
  } else if (state.scheduler === "exponential" && state.scheduler_factor !== 0.95) {
    cfg.scheduler_factor = state.scheduler_factor;
  }
  if (state.scheduler === "plateau") cfg.scheduler_patience = state.scheduler_patience;
  if (state.min_lr !== 1e-6) cfg.min_lr = state.min_lr;
  if (state.scheduler === "linear_warmup") cfg.warmup_epochs = state.warmup_epochs;
  if (state.scheduler === "step") cfg.step_size = state.step_size;
  if (state.scheduler === "multistep" && state.milestones) cfg.milestones = state.milestones;

  if (state.data_path) cfg.data_path = state.data_path;
  if (state.output_dir) cfg.output_dir = state.output_dir;
  if (state.resume) cfg.resume = state.resume;
  if (state.save_every !== 50) cfg.save_every = state.save_every;
  if (state.fresh) cfg.fresh = true;
  if (state.single_channel) cfg.single_channel = true;
  if (state.allow_pickle) cfg.allow_pickle = true;

  if (state.wandb) {
    cfg.wandb = true;
    if (state.wandb_watch) cfg.wandb_watch = true;
    if (state.project_name) cfg.project_name = state.project_name;
    if (state.run_name) cfg.run_name = state.run_name;
  }

  if (state.constraint && state.constraint.length > 0) {
    cfg.constraint = state.constraint;
    cfg.constraint_weight = state.constraint_weight;
    cfg.constraint_reduction = state.constraint_reduction;
  }
  if (state.constraint_file) {
    cfg.constraint_file = state.constraint_file;
    cfg.constraint_weight = state.constraint_weight;
    cfg.constraint_reduction = state.constraint_reduction;
  }

  for (const [k, v] of Object.entries(extraConfigKeys)) {
    if (cfg[k] === undefined && v !== null && v !== undefined && v !== "" && !(Array.isArray(v) && v.length === 0)) {
      cfg[k] = v;
    }
  }

  return cfg;
}

function updateCliPreview() {
  const cfgPath = (document.getElementById("configFilePath").value || "config.yaml").trim();
  const gpus = (document.getElementById("f_num_gpus").value || "").trim();
  const gpuFlag = gpus && parseInt(gpus, 10) > 0 ? ` --num_gpus ${parseInt(gpus, 10)}` : "";
  document.getElementById("cliPreview").textContent = `wavedl-train${gpuFlag} --config ${cfgPath}`;
}

function renderYamlPreview() {
  const c = buildEffectiveConfig();
  const fmtScalar = (v) => {
    if (typeof v === "boolean") return v ? "true" : "false";
    if (v === null || v === undefined) return "null";
    if (typeof v === "number") return Number.isFinite(v) ? String(v) : "null";
    const s = String(v);
    const reserved = /^(true|false|yes|no|on|off|y|n|null|none|nan|\.nan|inf|\.inf|\+\.inf|-\.inf)$/i.test(s);
    const safeUnquoted = /^[a-zA-Z0-9_./-]+$/.test(s) && (/^[a-zA-Z]/.test(s) || s.startsWith("./") || s.startsWith("../") || s.startsWith("/"));
    return safeUnquoted && !reserved ? s : JSON.stringify(s);
  };
  const fmtVal = (v) => Array.isArray(v) ? `[${v.map(fmtScalar).join(", ")}]` : fmtScalar(v);

  const groups = [
    ["MODEL", ["model", "pretrained", "import_modules"]],
    ["TRAINING HYPERPARAMETERS", ["batch_size", "lr", "epochs", "patience", "weight_decay", "grad_clip", "grad_accum_steps"]],
    ["LOSS FUNCTION", ["loss", "huber_delta", "loss_weights", "selection_metric"]],
    ["OPTIMIZER", ["optimizer", "momentum", "nesterov", "betas"]],
    ["LEARNING RATE SCHEDULER", ["scheduler", "scheduler_factor", "scheduler_patience", "min_lr", "warmup_epochs", "step_size", "milestones"]],
    ["CROSS-VALIDATION", ["cv", "cv_stratify", "cv_bins"]],
    ["DATA & OUTPUT", ["data_path", "output_dir", "cache_dir", "resume", "save_every", "fresh", "workers", "seed", "deterministic", "cache_validate", "single_channel", "allow_pickle"]],
    ["PERFORMANCE OPTIMIZATION", ["precision", "compile"]],
    ["LOGGING & MONITORING", ["wandb", "wandb_watch", "project_name", "run_name"]],
    ["PHYSICAL CONSTRAINTS", ["constraint", "constraint_file", "constraint_weight", "constraint_reduction"]]
  ];

  const used = new Set();
  const lines = [
    "# WaveDL Configuration",
    "# Generated by WaveDL Studio",
    ""
  ];
  for (const [title, keys] of groups) {
    const present = keys.filter(k => c[k] !== undefined && c[k] !== "" && !(Array.isArray(c[k]) && c[k].length === 0));
    if (present.length === 0) continue;
    const heading = title.toLowerCase().replace(/\b[a-z]/g, c => c.toUpperCase());
    lines.push(`# ── ${heading}`);
    for (const k of present) {
      used.add(k);
      lines.push(`${k}: ${fmtVal(c[k])}`);
    }
    lines.push("");
  }

  const extras = Object.keys(c).filter(k => !used.has(k) && !k.startsWith("_") && c[k] !== undefined && c[k] !== "" && !(Array.isArray(c[k]) && c[k].length === 0));
  if (extras.length > 0) {
    lines.push("# ── Additional Options");
    for (const k of extras) {
      lines.push(`${k}: ${fmtVal(c[k])}`);
    }
    lines.push("");
  }

  document.getElementById("yamlEditor").value = lines.join("\n").trimEnd() + "\n";
  updateCliPreview();
}

function syncFromForm() {
  document.querySelectorAll('input[inputmode="decimal"]').forEach(input => {
    input.value = input.value.replace(/,/g, ".");
  });
  readDOMToState();
  updateConditionalFields();
  updateSummaryAndBadges();
  renderYamlPreview();
}

function normalizeLoadedConfig(rawCfg) {
  const incoming = Object.assign({}, rawCfg || {});
  if (incoming.import !== undefined && incoming.import_modules === undefined) {
    incoming.import_modules = incoming.import;
    delete incoming.import;
  }
  if (typeof incoming.import_modules === "string") {
    incoming.import_modules = incoming.import_modules.split(",").map(s => s.trim()).filter(Boolean);
  }
  if (typeof incoming.constraint === "string") {
    incoming.constraint = incoming.constraint.split(/[;\n]+/).map(s => s.trim()).filter(Boolean);
  }
  if (typeof incoming.constraint_weight === "number") {
    incoming.constraint_weight = [incoming.constraint_weight];
  } else if (typeof incoming.constraint_weight === "string") {
    incoming.constraint_weight = incoming.constraint_weight
      .split(",")
      .map(s => parseFloat(s.trim()))
      .filter(Number.isFinite);
  }
  ["betas", "milestones", "loss_weights"].forEach(k => {
    if (Array.isArray(incoming[k])) {
      incoming[k] = incoming[k].join(",");
    }
  });

  const merged = JSON.parse(JSON.stringify(DEFAULTS));
  extraConfigKeys = {};
  for (const [k, v] of Object.entries(incoming)) {
    if (v === null || v === undefined) continue;
    if (Object.prototype.hasOwnProperty.call(DEFAULTS, k)) {
      merged[k] = v;
    } else if (!k.startsWith("_")) {
      extraConfigKeys[k] = v;
    }
  }
  if (merged.scheduler === "exponential" && incoming.scheduler_factor === undefined) {
    merged.scheduler_factor = 0.95;
  }

  const hasConstraints = (Array.isArray(merged.constraint) && merged.constraint.length > 0) || Boolean(merged.constraint_file);
  document.getElementById("toggle_constraints_enabled").checked = hasConstraints;
  return merged;
}

async function loadConfigFromFile(silent = false) {
  const path = document.getElementById("configFilePath").value.trim() || "config.yaml";
  try {
    const res = await fetch(`/api/config?path=${encodeURIComponent(path)}`);
    const data = await res.json();
    if (!res.ok) {
      if (!silent) showToast(data.error || "Could not load config", true);
      return;
    }
    state = normalizeLoadedConfig(data.config || {});
    applyStateToDOM();
    if (!silent) showToast(`Loaded ${path}`);
  } catch (e) {
    if (!silent) showToast(`Error loading config: ${e}`, true);
  }
}

async function saveConfigToFile() {
  const path = document.getElementById("configFilePath").value.trim() || "config.yaml";
  const rawYaml = document.getElementById("yamlEditor").value;
  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: postHeaders(),
      body: JSON.stringify({ path, raw_yaml: rawYaml })
    });
    const data = await res.json();
    if (!res.ok) {
      showToast(data.error || "Failed to save config", true);
      return false;
    }
    showToast(`Saved config to ${data.path}`);
    return true;
  } catch (e) {
    showToast(`Error saving config: ${e}`, true);
    return false;
  }
}

async function applyYamlToForm() {
  const rawYaml = document.getElementById("yamlEditor").value;
  try {
    const res = await fetch("/api/config/parse", {
      method: "POST",
      headers: postHeaders(),
      body: JSON.stringify({ raw_yaml: rawYaml })
    });
    if (res.status === 404) {
      const saved = await saveConfigToFile();
      if (saved) await loadConfigFromFile(true);
      return;
    }
    const data = await res.json();
    if (!res.ok) {
      showToast(data.error || "Invalid YAML", true);
      return;
    }
    state = normalizeLoadedConfig(data.config || {});
    applyStateToDOM();
    showToast("Synced YAML edits into form");
  } catch (e) {
    showToast(`Failed to parse YAML: ${e}`, true);
  }
}

function resetDefaults() {
  state = JSON.parse(JSON.stringify(DEFAULTS));
  extraConfigKeys = {};
  document.getElementById("toggle_constraints_enabled").checked = false;
  applyStateToDOM();
  showToast("Reset to WaveDL default configuration");
}

async function copyTextWithFallback(text, successMsg) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      showToast(successMsg);
      return;
    }
  } catch (_) {}
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand("copy");
    showToast(successMsg);
  } catch (_) {
    showToast("Could not copy automatically — select text and press Ctrl/Cmd+C", true);
  } finally {
    document.body.removeChild(ta);
  }
}

function copyYaml() {
  copyTextWithFallback(document.getElementById("yamlEditor").value, "Copied YAML to clipboard");
}

function copyCli() {
  copyTextWithFallback(document.getElementById("cliPreview").textContent, "Copied CLI command to clipboard");
}

async function startTraining() {
  if (state.fresh && state.resume) {
    showToast("Cannot enable both 'Start Fresh' and 'Resume From Checkpoint'.", true);
    return;
  }
  if (state.cv >= 2) {
    if (state.resume) {
      showToast("K-fold cross-validation (--cv) cannot be combined with --resume.", true);
      return;
    }
    if ((Array.isArray(state.constraint) && state.constraint.length > 0) || state.constraint_file) {
      showToast("K-fold cross-validation (--cv) does not support physical constraints.", true);
      return;
    }
    if (state.grad_accum_steps > 1) {
      showToast("K-fold cross-validation (--cv) does not support grad_accum_steps > 1.", true);
      return;
    }
  }

  const path = document.getElementById("configFilePath").value.trim() || "config.yaml";
  const rawYaml = document.getElementById("yamlEditor").value;
  const gpus = parseInt(document.getElementById("f_num_gpus").value, 10);

  setConsoleDrawer(true);
  try {
    const res = await fetch("/api/train/start", {
      method: "POST",
      headers: postHeaders(),
      body: JSON.stringify({
        path,
        raw_yaml: rawYaml,
        num_gpus: Number.isFinite(gpus) && gpus > 0 ? gpus : null
      })
    });
    const data = await res.json();
    if (!res.ok) {
      showToast(data.error || "Failed to start training", true);
      return;
    }
    logOffset = 0;
    activeJobId = data.job_id !== undefined ? data.job_id : null;
    document.getElementById("consoleOutput").textContent = "";
    showToast("Saved config & started wavedl-train!");
    checkTrainingStatus();
  } catch (e) {
    showToast(`Launch failed: ${e}`, true);
  }
}

async function startTest(mode) {
  const val = id => document.getElementById(id).value.trim();
  const payload = {
    mode,
    checkpoint: val("t_checkpoint"),
    data_path: val("t_data_path"),
    output_dir: val("t_output_dir")
  };
  if (Array.isArray(state.import_modules) && state.import_modules.length > 0) {
    payload.import_modules = state.import_modules;
  }
  if (state.allow_pickle) {
    payload.allow_pickle = true;
  }
  if (state.single_channel) {
    payload.input_channels = 1;
  }
  setConsoleDrawer(true);
  try {
    const res = await fetch("/api/test/start", {
      method: "POST",
      headers: postHeaders(),
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) {
      showToast(data.error || `Failed to start ${mode}`, true);
      return;
    }
    logOffset = 0;
    activeJobId = data.job_id !== undefined ? data.job_id : null;
    document.getElementById("consoleOutput").textContent = "";
    showToast(mode === "export" ? "Exporting model to ONNX..." : "Started wavedl-test!");
    checkTrainingStatus();
  } catch (e) {
    showToast(`Launch failed: ${e}`, true);
  }
}

async function stopTraining() {
  try {
    await fetch("/api/train/stop", {
      method: "POST",
      headers: postHeaders(),
      body: "{}"
    });
    showToast("Stopping training process...");
    checkTrainingStatus();
  } catch (e) {
    showToast(`Failed to stop: ${e}`, true);
  }
}

async function checkTrainingStatus() {
  if (pollTimer) clearTimeout(pollTimer);
  try {
    const res = await fetch(`/api/train/status?offset=${logOffset}`);
    const data = await res.json();
    const consoleEl = document.getElementById("consoleOutput");

    if (data.job_id !== undefined && activeJobId !== null && data.job_id !== activeJobId) {
      activeJobId = data.job_id;
      logOffset = 0;
      consoleEl.textContent = "";
      pollTimer = setTimeout(checkTrainingStatus, 50);
      return;
    }
    if (data.job_id !== undefined) {
      activeJobId = data.job_id;
    }

    if (data.logs && data.logs.length > 0) {
      if (logOffset === 0) consoleEl.textContent = "";
      let combined = consoleEl.textContent + data.logs.join("\n") + "\n";
      if (combined.length > 250000) {
        const cut = combined.indexOf("\n", combined.length - 200000);
        combined = cut >= 0 ? combined.slice(cut + 1) : combined.slice(-200000);
      }
      consoleEl.textContent = combined;
      consoleEl.scrollTop = consoleEl.scrollHeight;
      document.getElementById("consoleLatest").textContent = combined.trimEnd().split("\n").pop() || "Not started";
      logOffset = data.next_offset;
    }

    const badge = document.getElementById("trainStatusBadge");
    const runBtn = document.getElementById("runBtn");
    const stopBtn = document.getElementById("stopBtn");
    const elapsedEl = document.getElementById("trainElapsed");

    elapsedEl.textContent = data.elapsed !== null ? `(${data.elapsed}s)` : "";

    if (data.running) {
      if (!wasJobRunning) setConsoleDrawer(true);
      wasJobRunning = true;
      badge.className = "status-pill status-running";
      badge.textContent = `Running (PID ${data.pid})`;
      runBtn.style.display = "none";
      stopBtn.style.display = "inline-flex";
      pollTimer = setTimeout(checkTrainingStatus, 600);
    } else {
      wasJobRunning = false;
      runBtn.style.display = "inline-flex";
      stopBtn.style.display = "none";
      if (data.exit_code === 0) {
        badge.className = "status-pill status-ok";
        badge.textContent = "Completed";
      } else if (data.exit_code !== null && data.exit_code !== undefined) {
        badge.className = "status-pill status-err";
        badge.textContent = `Failed (${data.exit_code})`;
      } else {
        badge.className = "status-pill status-idle";
        badge.textContent = "Idle";
      }
    }
  } catch (_) {
    if (wasJobRunning) {
      pollTimer = setTimeout(checkTrainingStatus, 1500);
    }
  }
}

function clearConsole() {
  document.getElementById("consoleOutput").textContent = "";
  document.getElementById("consoleLatest").textContent = "Console cleared";
}

// File / Directory Browser Modal
let browserTargetInput = null;
let browserMode = "file";
let currentBrowseDir = "";
let parentBrowseDir = "";
let workspaceRootDir = "";

async function openBrowser(inputId, mode) {
  browserTargetInput = inputId;
  browserMode = mode;
  document.getElementById("browserTitle").textContent =
    mode === "dir" ? "Select Folder" : "Select File";
  document.getElementById("browserSelectDirBtn").style.display =
    mode === "dir" ? "inline-flex" : "none";
  document.getElementById("browserModal").classList.add("open");
  const curVal = document.getElementById(inputId).value.trim();
  await fetchBrowse(curVal || "");
}

function closeBrowser() {
  document.getElementById("browserModal").classList.remove("open");
}

async function fetchBrowse(dir) {
  const res = await fetch(`/api/browse?dir=${encodeURIComponent(dir)}`);
  const data = await res.json();
  currentBrowseDir = data.current_dir;
  parentBrowseDir = data.parent_dir;
  workspaceRootDir = data.workspace_root;
  document.getElementById("browserCurrentDir").textContent = currentBrowseDir;

  const list = document.getElementById("browserList");
  list.textContent = "";
  if (!data.entries || data.entries.length === 0) {
    const empty = document.createElement("div");
    empty.style.padding = "14px";
    empty.style.color = "var(--muted)";
    empty.textContent = "No files or folders found.";
    list.appendChild(empty);
    return;
  }
  data.entries.forEach(item => {
    const row = document.createElement("div");
    row.className = "file-row";
    const left = document.createElement("span");
    const right = document.createElement("span");
    right.style.fontSize = "12px";

    if (item.is_dir) {
      left.innerHTML = icon("folder");
      const strong = document.createElement("strong");
      strong.textContent = `${item.name}/`;
      left.appendChild(strong);
      right.style.color = "var(--muted)";
      right.textContent = "Open →";
      row.addEventListener("click", () => fetchBrowse(item.abs_path));
    } else if (browserMode === "dir") {
      left.innerHTML = icon("file");
      left.append(document.createTextNode(item.name));
      row.style.cursor = "default";
      row.style.opacity = "0.6";
    } else {
      left.innerHTML = icon("file");
      left.append(document.createTextNode(item.name));
      right.style.color = "var(--brand-dark)";
      right.style.fontWeight = "600";
      right.textContent = "Select";
      if (browserTargetInput === "configFilePath" && !/\.ya?ml$/i.test(item.name)) {
        row.style.opacity = "0.45";
        row.style.cursor = "default";
        right.textContent = "";
      } else {
        row.addEventListener("click", () => pickFile(item.path));
      }
    }
    row.appendChild(left);
    row.appendChild(right);
    list.appendChild(row);
  });
}

function browseUp() {
  if (parentBrowseDir) fetchBrowse(parentBrowseDir);
}

function browseRoot() {
  if (workspaceRootDir) fetchBrowse(workspaceRootDir);
}

function pickFile(path) {
  if (browserTargetInput) {
    document.getElementById(browserTargetInput).value = path;
    if (browserTargetInput === "configFilePath") loadConfigFromFile();
    else if (browserTargetInput === "f_data_path") onDataPathChange();
    else if (!browserTargetInput.startsWith("t_")) syncFromForm();
  }
  closeBrowser();
}

function chooseCurrentDir() {
  if (browserTargetInput && currentBrowseDir) {
    const sep = currentBrowseDir.includes("\\") ? "\\" : "/";
    const prefix = workspaceRootDir.replace(/[\\/]+$/, "") + sep;
    const rel = currentBrowseDir.startsWith(prefix)
      ? currentBrowseDir.slice(prefix.length)
      : (currentBrowseDir === workspaceRootDir ? "." : currentBrowseDir);
    document.getElementById(browserTargetInput).value = rel;
    if (!browserTargetInput.startsWith("t_")) syncFromForm();
  }
  closeBrowser();
}

function showToast(msg, isErr = false) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.style.background = isErr ? "var(--danger)" : "var(--brand-dark)";
  t.style.display = "block";
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.style.display = "none"; }, isErr ? 5000 : 2600);
}

window.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
    e.preventDefault();
    saveConfigToFile();
  } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "o") {
    e.preventDefault();
    loadConfigFromFile();
  } else if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    startTraining();
  } else if (e.key === "Escape") {
    closeBrowser();
  }
});

init();
