import os
import tempfile
from datetime import datetime

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

st.set_page_config(
    page_title="Veritas | Audio Forensics Console",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap');

:root {
    --bg-void: #060606;
    --bg-panel: #0f0d0d;
    --bg-raised: #181313;
    --line: #2a1a1a;
    --line-bright: #4a2424;
    --red: #e2392f;
    --red-dim: #8a2620;
    --green: #4fae6e;
    --text: #f2e9e6;
    --text-dim: #9c8a86;
    --text-faint: #5e4f4c;
}

html, body, .stApp {
    background-color: var(--bg-void) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}

p, span, label, div, li { color: var(--text); }

.mono { font-family: 'JetBrains Mono', monospace; }

/* ---- Top masthead ---- */
.masthead {
    border-bottom: 1px solid var(--line);
    padding: 0 0 18px 0;
    margin-bottom: 28px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}
.masthead-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.masthead-title .dot {
    color: var(--red);
    font-size: 1.4rem;
    animation: pulse 2.2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.25; }
}
.masthead-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-faint);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 2px;
}
.masthead-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-dim);
    text-align: right;
    line-height: 1.6;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background-color: var(--bg-panel) !important;
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    background-color: transparent;
    border-radius: 0;
    padding: 10px 18px;
}
.stTabs [aria-selected="true"] {
    color: var(--red) !important;
    border-bottom: 2px solid var(--red) !important;
}

/* ---- Cards / panels ---- */
.panel {
    background: var(--bg-panel);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--text-faint);
    margin-bottom: 10px;
}

/* ---- Verdict blocks ---- */
.verdict {
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 28px;
    position: relative;
    overflow: hidden;
}
.verdict.fake { border-color: var(--red-dim); background: linear-gradient(180deg, rgba(226,57,47,0.08), transparent); }
.verdict.real { border-color: #2a4a36; background: linear-gradient(180deg, rgba(79,174,110,0.08), transparent); }
.verdict-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--text-faint);
}
.verdict-headline {
    font-size: 2.6rem;
    font-weight: 700;
    margin: 6px 0 4px;
    letter-spacing: -0.02em;
}
.verdict.fake .verdict-headline { color: var(--red); }
.verdict.real .verdict-headline { color: var(--green); }
.verdict-desc { color: var(--text-dim); font-size: 0.92rem; max-width: 480px; }

/* ---- Metric readouts ---- */
.metric-grid { display: flex; gap: 14px; margin-top: 18px; flex-wrap: wrap; }
.metric-box {
    flex: 1; min-width: 140px;
    background: var(--bg-raised);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 14px 16px;
}
.metric-box .k {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-faint);
}
.metric-box .v {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    margin-top: 4px;
}

/* ---- Bars ---- */
.bar-row { margin-top: 14px; }
.bar-row .lbl {
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: var(--text-dim);
    margin-bottom: 4px;
}
.bar-track { height: 8px; background: var(--bg-raised); border-radius: 2px; overflow: hidden; border: 1px solid var(--line); }
.bar-fill { height: 100%; }
.bar-fill.red { background: var(--red); }
.bar-fill.green { background: var(--green); }

/* ---- File uploader ---- */
[data-testid="stFileUploader"] {
    background: var(--bg-panel);
    border: 1px dashed var(--line-bright);
    border-radius: 4px;
    padding: 6px;
}
[data-testid="stFileUploaderDropzoneInstructions"] { color: var(--text-dim); }

/* ---- Buttons ---- */
.stButton button, .stDownloadButton button {
    background: transparent !important;
    border: 1px solid var(--line-bright) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.stButton button:hover, .stDownloadButton button:hover {
    border-color: var(--red) !important;
    color: var(--red) !important;
}

/* ---- Expander ---- */
[data-testid="stExpander"] {
    background: var(--bg-panel);
    border: 1px solid var(--line);
    border-radius: 4px;
}

/* ---- Slider ---- */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: var(--red) !important; }

/* ---- Alerts ---- */
.stAlert { background-color: var(--bg-panel) !important; border: 1px solid var(--line-bright) !important; border-radius: 4px !important; }

/* ---- Dataframe / table ---- */
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 4px; }

/* ---- Status pill ---- */
.pill {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: 100px;
    border: 1px solid;
}
.pill.fake { color: var(--red); border-color: var(--red-dim); }
.pill.real { color: var(--green); border-color: #2a4a36; }

hr { border-color: var(--line) !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MODEL DEFINITION
# ---------------------------------------------------------------------------
class Settings:
    sample_rate = 16000
    duration = 4
    n_mels = 128
    n_fft = 1024
    hop_length = 256
    f_min = 20
    f_max = 8000

    max_samples = sample_rate * duration
    time_frames = max_samples // hop_length + 1

    cnn_channels = [1, 32, 64, 128]
    cnn_dropout = 0.2

    d_model = 128
    n_heads = 8
    n_layers = 4
    ff_dim = 512
    tf_dropout = 0.1

    model_path = "best_model.pt"


S = Settings()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=(2, 2)):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.MaxPool2d(pool),
            nn.Dropout2d(S.cnn_dropout),
        )

    def forward(self, x):
        return self.layers(x)


class SpectrogramEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch = S.cnn_channels
        self.block1 = ConvBlock(ch[0], ch[1], pool=(2, 2))
        self.block2 = ConvBlock(ch[1], ch[2], pool=(2, 2))
        self.block3 = ConvBlock(ch[2], ch[3], pool=(2, 1))
        self.proj = nn.Linear(ch[3] * (S.n_mels // 8), S.d_model)
        self.norm = nn.LayerNorm(S.d_model)

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        batch, channels, height, width = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, width, channels * height)
        return self.norm(self.proj(x))


class TemporalEncoder(nn.Module):
    def __init__(self, max_len=500):
        super().__init__()
        self.position_embed = nn.Embedding(max_len, S.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=S.d_model,
            nhead=S.n_heads,
            dim_feedforward=S.ff_dim,
            dropout=S.tf_dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=S.n_layers, norm=nn.LayerNorm(S.d_model)
        )

    def forward(self, x):
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.encoder(x + self.position_embed(positions))


class AttentionClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.attn_score = nn.Linear(S.d_model, 1)
        self.classifier = nn.Sequential(
            nn.Linear(S.d_model, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        weights = torch.softmax(self.attn_score(x), dim=1)
        pooled = (weights * x).sum(dim=1)
        return self.classifier(pooled)


class AudioAuthenticityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpectrogramEncoder()
        self.temporal = TemporalEncoder()
        self.head = AttentionClassifier()

    def forward(self, x):
        return self.head(self.temporal(self.encoder(x)))


# Maps old checkpoint key prefixes (from a previous refactor of the model
# class) to the current module names defined above, so existing checkpoints
# still load without retraining.
KEY_RENAME_MAP = [
    ("cnn.b1.net.", "encoder.block1.layers."),
    ("cnn.b2.net.", "encoder.block2.layers."),
    ("cnn.b3.net.", "encoder.block3.layers."),
    ("cnn.proj.", "encoder.proj."),
    ("cnn.norm.", "encoder.norm."),
    ("transformer.pos.", "temporal.position_embed."),
    ("transformer.tf.", "temporal.encoder."),
    ("head.attn.", "head.attn_score."),
    ("head.mlp.", "head.classifier."),
]


def remap_state_dict_keys(state_dict: dict) -> dict:
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in KEY_RENAME_MAP:
            if new_key.startswith(old_prefix):
                new_key = new_prefix + new_key[len(old_prefix):]
                break
        remapped[new_key] = value
    return remapped


@st.cache_resource
def load_model():
    if not os.path.exists(S.model_path):
        return None

    model = AudioAuthenticityModel().to(DEVICE)
    raw_state_dict = torch.load(S.model_path, map_location=DEVICE, weights_only=True)

    try:
        model.load_state_dict(raw_state_dict)
    except RuntimeError:
        remapped_state_dict = remap_state_dict_keys(raw_state_dict)
        model.load_state_dict(remapped_state_dict)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# AUDIO PROCESSING
# ---------------------------------------------------------------------------
def bytes_to_waveform(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        samples, sr = librosa.load(tmp_path, sr=None, mono=True)
    finally:
        os.unlink(tmp_path)
    return samples, sr


def waveform_to_melspec(samples: np.ndarray, sr: int):
    mel_transform = T.MelSpectrogram(
        sample_rate=S.sample_rate,
        n_fft=S.n_fft,
        hop_length=S.hop_length,
        n_mels=S.n_mels,
        f_min=S.f_min,
        f_max=S.f_max,
    )
    db_transform = T.AmplitudeToDB(top_db=80)

    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)

    if sr != S.sample_rate:
        waveform = T.Resample(sr, S.sample_rate)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    length = waveform.shape[1]
    if length < S.max_samples:
        waveform = F.pad(waveform, (0, S.max_samples - length))
    else:
        waveform = waveform[:, :S.max_samples]

    mel = db_transform(mel_transform(waveform))
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.unsqueeze(0), mel.squeeze().numpy()


def run_inference(model, mel_tensor, threshold):
    mel_tensor = mel_tensor.to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(mel_tensor), dim=1)[0].cpu().numpy()
    genuine_prob = float(probs[0])
    deepfake_prob = float(probs[1])
    label = "Deepfake" if deepfake_prob >= threshold else "Genuine"
    confidence = deepfake_prob if label == "Deepfake" else genuine_prob
    return genuine_prob, deepfake_prob, label, confidence


def render_waveform(samples: np.ndarray, sr: int, accent: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 1.8))
    fig.patch.set_facecolor("#060606")
    ax.set_facecolor("#060606")
    t = np.linspace(0, len(samples) / sr, num=len(samples))
    ax.plot(t, samples, color=accent, linewidth=0.6)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def render_spectrogram(mel_np: np.ndarray, cmap="inferno") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#060606")
    ax.set_facecolor("#060606")

    img = ax.imshow(mel_np, aspect="auto", origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_xlabel("Time frames", color="#9c8a86", fontsize=9)
    ax.set_ylabel("Mel bins", color="#9c8a86", fontsize=9)
    ax.tick_params(colors="#9c8a86", labelsize=8)

    for spine in ax.spines.values():
        spine.set_edgecolor("#2a1a1a")

    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color="#9c8a86", labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#9c8a86")

    plt.tight_layout()
    return fig


def render_segment_confidence(model, samples, sr, threshold, n_segments=8):
    """Slide a window over the clip and score each segment independently."""
    seg_len = int(sr * S.duration)
    total = len(samples)
    if total < seg_len:
        samples = np.pad(samples, (0, seg_len - total))
        total = len(samples)

    starts = np.linspace(0, max(total - seg_len, 0), num=n_segments).astype(int)
    scores = []
    for start in starts:
        seg = samples[start:start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        mel_tensor, _ = waveform_to_melspec(seg, sr)
        _, deepfake_prob, _, _ = run_inference(model, mel_tensor, threshold)
        scores.append(deepfake_prob)
    return scores


def render_segment_chart(scores, threshold) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 1.6))
    fig.patch.set_facecolor("#060606")
    ax.set_facecolor("#060606")
    x = np.arange(len(scores))
    colors = ["#e2392f" if s >= threshold else "#4fae6e" for s in scores]
    ax.bar(x, scores, color=colors, width=0.6)
    ax.axhline(threshold, color="#5e4f4c", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}" for i in x], color="#9c8a86", fontsize=8)
    ax.tick_params(colors="#9c8a86", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a1a1a")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------------
# MASTHEAD
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="masthead">
    <div>
        <p class="masthead-title"><span class="dot">◉</span> VERITAS</p>
        <p class="masthead-sub">Audio authenticity forensics console</p>
    </div>
    <div class="masthead-meta">
        ENGINE&nbsp;&nbsp;CNN+TRANSFORMER<br>
        DEVICE&nbsp;&nbsp;{str(DEVICE).upper()}<br>
        SESSION&nbsp;&nbsp;{datetime.now().strftime('%H:%M:%S')}
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR — CONTROL PANEL
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="panel-label">Control panel</p>', unsafe_allow_html=True)

    model = load_model()

    if model is None:
        st.error("Model weights `best_model.pt` not found in app directory.")
        status_pill = '<span class="pill fake">offline</span>'
    else:
        status_pill = '<span class="pill real">ready</span>'

    st.markdown(f"**Model status** {status_pill}", unsafe_allow_html=True)
    st.markdown("---")

    threshold = st.slider(
        "Sensitivity threshold",
        min_value=0.05, max_value=0.95, value=0.50, step=0.01,
        help="Probability above which a clip is flagged as Deepfake. Lower = more sensitive (more flags)."
    )
    st.caption(f"Clips scoring ≥ `{threshold:.2f}` deepfake probability are flagged.")

    st.markdown("---")
    st.markdown('<p class="panel-label">Display options</p>', unsafe_allow_html=True)
    show_waveform = st.checkbox("Show waveform", value=True)
    show_spectrogram = st.checkbox("Show mel-spectrogram", value=True)
    show_segments = st.checkbox("Show segment-wise breakdown", value=False,
                                 help="Splits the clip into windows and scores each independently.")
    spec_cmap = st.selectbox("Spectrogram colormap", ["inferno", "magma", "viridis", "cividis"], index=0)

    st.markdown("---")
    st.markdown('<p class="panel-label">Session</p>', unsafe_allow_html=True)
    st.metric("Clips analysed", len(st.session_state.history))
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_detect, tab_batch, tab_history, tab_about = st.tabs(
    ["Detect", "Batch", "History", "About"]
)


# ---------------------------------------------------------------------------
# TAB: DETECT (single file, full analysis)
# ---------------------------------------------------------------------------
with tab_detect:
    if model is None:
        st.warning("Place `best_model.pt` next to `app.py` to enable detection.")
    else:
        left, right = st.columns([1, 1.4])

        uploaded = None
        with left:
            st.markdown('<div class="panel"><p class="panel-label">Input</p>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Drop a recording",
                type=["wav", "mp3", "flac", "ogg"],
                help="WAV, MP3, FLAC, OGG supported.",
                key="single_upload",
            )
            if uploaded is not None:
                st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")
                st.caption(f"`{uploaded.name}` · {uploaded.size / 1024:.1f} KB")
            st.markdown('</div>', unsafe_allow_html=True)

        label = None
        samples = sr = mel_np = None

        with right:
            if uploaded is None:
                st.markdown(
                    '<div class="panel"><p class="panel-label">Result</p>'
                    '<p style="color:var(--text-faint); font-size:0.9rem;">'
                    'Awaiting input — drop a recording on the left to begin analysis.'
                    '</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("Running forensic analysis..."):
                    try:
                        file_bytes = uploaded.read()
                        samples, sr = bytes_to_waveform(file_bytes)
                        mel_tensor, mel_np = waveform_to_melspec(samples, sr)
                        genuine_prob, deepfake_prob, label, confidence = run_inference(
                            model, mel_tensor, threshold
                        )
                    except Exception as e:
                        st.error(f"Error processing audio: {e}")

                if label is not None:
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "file": uploaded.name,
                        "label": label,
                        "genuine": genuine_prob,
                        "deepfake": deepfake_prob,
                        "confidence": confidence,
                        "threshold": threshold,
                    })

                    verdict_class = "fake" if label == "Deepfake" else "real"
                    headline = "Deepfake detected" if label == "Deepfake" else "Genuine recording"
                    desc = (
                        "Spectral and temporal patterns are consistent with AI-generated speech synthesis."
                        if label == "Deepfake" else
                        "Spectral and temporal patterns are consistent with natural human speech."
                    )

                    st.markdown(f"""
                    <div class="verdict {verdict_class}">
                        <p class="verdict-tag">Detection result</p>
                        <p class="verdict-headline">{headline}</p>
                        <p class="verdict-desc">{desc}</p>
                        <div class="metric-grid">
                            <div class="metric-box">
                                <div class="k">Confidence</div>
                                <div class="v">{confidence*100:.1f}%</div>
                            </div>
                            <div class="metric-box">
                                <div class="k">Threshold</div>
                                <div class="v">{threshold:.2f}</div>
                            </div>
                            <div class="metric-box">
                                <div class="k">Duration</div>
                                <div class="v">{len(samples)/sr:.1f}s</div>
                            </div>
                        </div>
                        <div class="bar-row">
                            <div class="lbl"><span>Genuine</span><span>{genuine_prob*100:.2f}%</span></div>
                            <div class="bar-track"><div class="bar-fill green" style="width:{genuine_prob*100:.2f}%"></div></div>
                        </div>
                        <div class="bar-row">
                            <div class="lbl"><span>Deepfake</span><span>{deepfake_prob*100:.2f}%</span></div>
                            <div class="bar-track"><div class="bar-fill red" style="width:{deepfake_prob*100:.2f}%"></div></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        if uploaded is not None and label is not None:
            st.markdown("---")
            accent = "#e2392f" if label == "Deepfake" else "#4fae6e"

            if show_waveform:
                st.markdown('<p class="panel-label">Waveform</p>', unsafe_allow_html=True)
                fig = render_waveform(samples, sr, accent)
                st.pyplot(fig)
                plt.close(fig)

            if show_spectrogram:
                st.markdown('<p class="panel-label">Mel-spectrogram</p>', unsafe_allow_html=True)
                fig = render_spectrogram(mel_np, cmap=spec_cmap)
                st.pyplot(fig)
                plt.close(fig)

            if show_segments:
                st.markdown('<p class="panel-label">Segment-wise deepfake probability</p>', unsafe_allow_html=True)
                with st.spinner("Scoring segments..."):
                    scores = render_segment_confidence(model, samples, sr, threshold)
                fig = render_segment_chart(scores, threshold)
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    "Each bar (S1-S8) is an independently scored window across the clip's duration. "
                    "Bars above the dashed line exceed the sensitivity threshold."
                )


# ---------------------------------------------------------------------------
# TAB: BATCH (multi-file)
# ---------------------------------------------------------------------------
with tab_batch:
    if model is None:
        st.warning("Place `best_model.pt` next to `app.py` to enable detection.")
    else:
        st.markdown('<div class="panel"><p class="panel-label">Batch input</p>', unsafe_allow_html=True)
        batch_files = st.file_uploader(
            "Drop multiple recordings",
            type=["wav", "mp3", "flac", "ogg"],
            accept_multiple_files=True,
            key="batch_upload",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if batch_files:
            run_batch = st.button(f"Analyse {len(batch_files)} file(s)")

            if run_batch:
                results = []
                progress = st.progress(0)
                status = st.empty()

                for i, f in enumerate(batch_files):
                    status.caption(f"Processing `{f.name}`...")
                    try:
                        samples, sr = bytes_to_waveform(f.read())
                        mel_tensor, _ = waveform_to_melspec(samples, sr)
                        genuine_prob, deepfake_prob, label, confidence = run_inference(
                            model, mel_tensor, threshold
                        )
                        results.append({
                            "File": f.name,
                            "Verdict": label,
                            "Confidence": f"{confidence*100:.1f}%",
                            "Genuine %": f"{genuine_prob*100:.2f}",
                            "Deepfake %": f"{deepfake_prob*100:.2f}",
                            "Duration (s)": f"{len(samples)/sr:.1f}",
                        })
                        st.session_state.history.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "file": f.name,
                            "label": label,
                            "genuine": genuine_prob,
                            "deepfake": deepfake_prob,
                            "confidence": confidence,
                            "threshold": threshold,
                        })
                    except Exception:
                        results.append({
                            "File": f.name,
                            "Verdict": "Error",
                            "Confidence": "-",
                            "Genuine %": "-",
                            "Deepfake %": "-",
                            "Duration (s)": "-",
                        })
                    progress.progress((i + 1) / len(batch_files))

                status.empty()
                progress.empty()

                n_fake = sum(1 for r in results if r["Verdict"] == "Deepfake")
                n_real = sum(1 for r in results if r["Verdict"] == "Genuine")

                st.markdown(f"""
                <div class="metric-grid">
                    <div class="metric-box"><div class="k">Total</div><div class="v">{len(results)}</div></div>
                    <div class="metric-box"><div class="k">Flagged deepfake</div><div class="v" style="color:var(--red)">{n_fake}</div></div>
                    <div class="metric-box"><div class="k">Genuine</div><div class="v" style="color:var(--green)">{n_real}</div></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(results, use_container_width=True, hide_index=True)

                csv_data = "File,Verdict,Confidence,Genuine %,Deepfake %,Duration (s)\n"
                for r in results:
                    csv_data += ",".join(str(v) for v in r.values()) + "\n"

                st.download_button(
                    "Download report (CSV)",
                    data=csv_data,
                    file_name=f"veritas_batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        else:
            st.caption("Upload two or more files to run a batch scan.")


# ---------------------------------------------------------------------------
# TAB: HISTORY
# ---------------------------------------------------------------------------
with tab_history:
    if not st.session_state.history:
        st.markdown(
            '<div class="panel"><p class="panel-label">History</p>'
            '<p style="color:var(--text-faint); font-size:0.9rem;">'
            'No clips analysed yet this session. Results from Detect and Batch tabs appear here.'
            '</p></div>',
            unsafe_allow_html=True,
        )
    else:
        n_total = len(st.session_state.history)
        n_fake = sum(1 for h in st.session_state.history if h["label"] == "Deepfake")
        n_real = n_total - n_fake

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-box"><div class="k">Total analysed</div><div class="v">{n_total}</div></div>
            <div class="metric-box"><div class="k">Deepfake</div><div class="v" style="color:var(--red)">{n_fake}</div></div>
            <div class="metric-box"><div class="k">Genuine</div><div class="v" style="color:var(--green)">{n_real}</div></div>
        </div>
        <br>
        """, unsafe_allow_html=True)

        table_rows = []
        for h in reversed(st.session_state.history):
            table_rows.append({
                "Time": h["time"],
                "File": h["file"],
                "Verdict": h["label"],
                "Confidence": f"{h['confidence']*100:.1f}%",
                "Genuine %": f"{h['genuine']*100:.2f}",
                "Deepfake %": f"{h['deepfake']*100:.2f}",
                "Threshold": f"{h['threshold']:.2f}",
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        csv_data = "Time,File,Verdict,Confidence,Genuine %,Deepfake %,Threshold\n"
        for r in table_rows:
            csv_data += ",".join(str(v) for v in r.values()) + "\n"
        st.download_button(
            "Download session log (CSV)",
            data=csv_data,
            file_name=f"veritas_session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# TAB: ABOUT
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("""
    <div class="panel">
        <p class="panel-label">Pipeline</p>
        <p style="color:var(--text-dim); font-size:0.92rem; line-height:1.7;">
        Audio is resampled to 16 kHz, converted to a 128-bin log-mel spectrogram,
        and normalised. A convolutional encoder extracts local time-frequency
        features, which a 4-layer transformer encodes across the full clip.
        An attention-pooled classifier head produces a genuine / deepfake
        probability pair.
        </p>
    </div>
    <div class="panel">
        <p class="panel-label">Model configuration</p>
        <div class="metric-grid">
            <div class="metric-box"><div class="k">Sample rate</div><div class="v">16 kHz</div></div>
            <div class="metric-box"><div class="k">Mel bins</div><div class="v">128</div></div>
            <div class="metric-box"><div class="k">Clip length</div><div class="v">4s</div></div>
            <div class="metric-box"><div class="k">Transformer layers</div><div class="v">4</div></div>
            <div class="metric-box"><div class="k">Attention heads</div><div class="v">8</div></div>
            <div class="metric-box"><div class="k">d_model</div><div class="v">128</div></div>
        </div>
    </div>
    <div class="panel">
        <p class="panel-label">Notes</p>
        <p style="color:var(--text-dim); font-size:0.92rem; line-height:1.7;">
        Clips longer than 4 seconds are scored on their first 4 seconds in
        Detect and Batch modes, unless segment-wise breakdown is enabled,
        which scores multiple windows across the full duration independently.
        The sensitivity threshold controls the deepfake-probability cutoff
        used to label a clip — lower thresholds flag more clips as deepfake.
        </p>
    </div>
    """, unsafe_allow_html=True)
