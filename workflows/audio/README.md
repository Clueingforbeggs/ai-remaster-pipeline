# Create Audio Track (music + sound effects)

This phase generates a soundtrack for a silent film and muxes it onto the latest render
**without re-encoding the video** (`-c:v copy`). It runs after Recomposition and before
Upscaling; when Upscaling is enabled, FlashVSR carries the new audio track through via its
existing audio-mux step.

Unlike the other phases, the ComfyUI graphs are built in code (see
`scripts/audio_models.py`) rather than loaded from JSON, so there are no `.json` files to
install here. The script discovers each node's inputs via ComfyUI's `/object_info`, so model
filenames are auto-selected from whatever you have installed.

## Models — downloaded automatically

Like the other phases, the models are fetched on demand the first time **Create Audio Track**
runs (`scripts/dependency_manager.py` → `ensure_audio_models`), and can be prefetched with
`install_windows.ps1 -DownloadModels`. See `docs/installer-model-sources.md` for the exact
repos/filenames. Audio models download in **soft mode**: a failure logs guidance and continues
rather than aborting the stage.

### Music — Stable Audio Open (ComfyUI core)
No custom nodes needed; these ship with ComfyUI:
`CheckpointLoaderSimple`, `CLIPTextEncode`, `EmptyLatentAudio`, `KSampler`,
`VAEDecodeAudio`, `SaveAudio`.

Auto-downloaded to `ComfyUI/models/checkpoints/stable_audio_open_1.0.safetensors`.
**This is a gated Hugging Face model** — accept the licence at
https://huggingface.co/stabilityai/stable-audio-open-1.0 and run `hf auth login` (or set
`HF_TOKEN`) so the download can authenticate; otherwise place the file there manually.
Override the filename on the Audio tab ("Stable Audio checkpoint") if yours differs.

### Sound effects — MMAudio (video → synchronized audio)
Install the custom node (the only manual install for this phase):
- `ComfyUI-MMAudio` → https://github.com/kijai/ComfyUI-MMAudio
  (provides `MMAudioModelLoader`, `MMAudioFeatureUtilsLoader`, `MMAudioSampler`)

Plus `ComfyUI-VideoHelperSuite` (already used by outpaint/upscale) for `VHS_LoadVideo`.

The MMAudio weights (model + VAE + Synchformer + CLIP) auto-download into
`ComfyUI/models/mmaudio/`. MMAudio analyses a low-resolution proxy — the short side is capped
at **384px** by default, since higher resolution only increases processing time without
improving the audio.

### Captioning (optional, recommended) — local Qwen-VL
The music score is mood-matched per scene using a local Qwen-VL caption node. Set the node's
class name in the Audio tab field **"Qwen-VL caption node (advanced)"** (e.g. a
`Qwen2_VL`/`QwenVL` captioning node from a vision pack). If the field is left blank or the
node is unavailable, captioning is skipped and the **Music style hint** (or a sensible
default) is used for every cue. A `ShowText|pysssss` node, if installed, is used to capture
the caption text from the graph.

## How the stems are assembled
- **Music**: scenes are detected from colour-histogram jumps (min/max cue length derived from
  "Music cue seconds"); one Stable Audio cue per scene, trimmed/padded to the exact scene
  length with short fades, then concatenated → a full-length music stem.
- **SFX**: the video is split into "SFX chunk seconds" windows; each is downscaled to a proxy
  and fed to MMAudio; the per-window audio is concatenated → a full-length effects stem.
- **Mix**: music is ducked under the effects via `sidechaincompress`, with per-stem gain set
  by "Music level (dB)" / "Sound effects level (dB)", then muxed onto the video as AAC.
