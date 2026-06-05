This folder contains ARP's bundled Qwen Image Edit workflow.

`Image Edit (Qwen 2511).json` is the default workflow for colour reference
generation and outpaint guide-frame generation. ARP should be usable without
depending on a matching workflow already being present in the user's ComfyUI
blueprints folder.

`Image Edit Inpaint (Qwen 2511).json` is the default masked-edit workflow used
by the advanced reference/guide editor when a SAM2/brush/wand mask is present.
It uses the same Qwen Image Edit model stack plus ComfyUI's latent
`SetLatentNoiseMask` path.

The wrapper can patch arbitrary node IDs for:
- load image
- mask image
- prompt text
- save image prefix

User-provided ComfyUI blueprints are still supported as overrides through the
References workflow setting.

See `docs/qwen-image-edit-workflow.md`.
