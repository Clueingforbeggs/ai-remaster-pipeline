from __future__ import annotations

import copy
import csv
import json
import shutil
import tempfile
import threading
import time
import urllib.request
import unittest
import sys
import argparse
import importlib.util
import os
import zipfile
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import comfy_api  # noqa: E402
import colorize_video  # noqa: E402
import generate_single_reference  # noqa: E402
import openai_generate_reference  # noqa: E402
import outpaint_video  # noqa: E402
import prepare_outpaint_input  # noqa: E402
import qwen_colorize_references  # noqa: E402

from ai_remaster_gui import app
from ai_remaster_gui import config
from ai_remaster_gui import server


class GuiSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._settings = copy.deepcopy(app.APP.settings)

    def tearDown(self) -> None:
        app.APP.settings = self._settings

    def test_source_resolver_accepts_ascii_pipe_for_full_width_pipe_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            real = folder / "King Kong Scene Pack ｜ King Kong [0JgMh4I2UjY].mp4"
            real.write_bytes(b"not a real video")
            typed = folder / "King Kong Scene Pack | King Kong [0JgMh4I2UjY].mp4"

            self.assertEqual(app.resolve_video_source(str(typed)), real)

    def test_deterministic_outpaint_output_path_uses_selected_source(self) -> None:
        app.APP.settings.setdefault("outpaint", {}).update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )

        output = app.outpaint_output_for("input/My Source.mp4", "16:9", "720")

        self.assertEqual(output, "intermediate/outpainted/My_Source_16x9_1280x704_outpainted.mp4")

    def test_outpaint_ltx_working_paths_use_model_safe_size(self) -> None:
        app.APP.settings.setdefault("outpaint", {}).update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )

        prepared = app.outpaint_prepared_for("input/My Source.mp4", app.APP.settings["outpaint"])
        manifest = app.outpaint_chunk_manifest_for("input/My Source.mp4", app.APP.settings["outpaint"])

        self.assertEqual(prepared.name, "My Source_1280x704_from1280x720_lifted.mp4")
        self.assertTrue(manifest.endswith("My_Source_16x9_1280x704_chunks.csv"))
        self.assertEqual(app.outpaint_output_for("input/My Source.mp4", "16:9", "720"), "intermediate/outpainted/My_Source_16x9_1280x704_outpainted.mp4")

    def test_source_height_outpaint_option_uses_video_height(self) -> None:
        app.APP.settings.setdefault("outpaint", {}).update(
            {
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )
        with mock.patch.object(server, "video_metrics", return_value={"height": 480}):
            output = app.outpaint_output_for("input/My Source.mp4", "16:9", "source")

        self.assertEqual(output, "intermediate/outpainted/My_Source_16x9_854x480_outpainted.mp4")

    def test_outpaint_source_crop_preserves_original_source_scale(self) -> None:
        args = argparse.Namespace(
            delivery_width=1280,
            delivery_height=720,
            crop_left=0,
            crop_right=0,
            crop_top=270,
            crop_bottom=270,
            black_lift=0.018,
            gamma=1.06,
        )
        info = {"width": 1440, "height": 1080, "fps": 24.0}

        self.assertEqual(prepare_outpaint_input.source_placement_size(args, info, 1280, 704), (960, 360, 960, 352))

        filter_text = prepare_outpaint_input.build_filter(args, info, 1280, 704)

        self.assertIn("crop=w=1440:h=540:x=0:y=270,scale=w=960:h=360:flags=lanczos,scale=w=960:h=352:flags=lanczos", filter_text)
        self.assertNotIn("force_original_aspect_ratio=decrease", filter_text)

    def test_portable_comfy_parent_resolves_to_inner_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            portable = Path(tmp_text)
            inner = portable / "ComfyUI"
            inner.mkdir()
            (inner / "main.py").write_text("# comfy\n", encoding="utf-8")

            self.assertEqual(config.resolve_comfy_dir(str(portable)), inner)

    def test_required_comfy_workflows_are_bundled(self) -> None:
        outpaint = app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json"
        qwen = app.ROOT / "workflows" / "qwen_image_edit" / "Image Edit (Qwen 2511).json"

        for workflow in (outpaint, qwen):
            with self.subTest(workflow=workflow.name):
                self.assertTrue(workflow.exists(), f"Missing bundled workflow: {workflow}")
                json.loads(workflow.read_text(encoding="utf-8-sig"))

        self.assertEqual(server.default_qwen_workflow({}), app.rel(qwen))
        self.assertEqual(server.qwen_workflow_for({"workflow": "D:/missing/blueprints/Qwen Custom.json"}, {}), app.rel(qwen))
        self.assertEqual(
            server.qwen_workflow_for(
                {
                    "workflow": (
                        "D:/ComfyUI/venv/Lib/site-packages/"
                        "comfyui_workflow_templates_media_image/templates/image_qwen_image_edit_2511.json"
                    )
                },
                {},
            ),
            app.rel(qwen),
        )

    def test_required_custom_nodes_are_bundled(self) -> None:
        required = {
            "ComfyUI-LTXVideo": ("LTXVImgToVideoConditionOnly", "LTXAddVideoICLoRAGuide", "LTXVPreprocess"),
            "ComfyUI-GGUF": ("UnetLoaderGGUF",),
            "ComfyUI-VideoHelperSuite": ("VHS_LoadVideo", "VHS_VideoCombine"),
            "reference-video-colorization": ("DeepExColorVideoNode", "ColorMNetVideo"),
        }
        vendor_root = app.ROOT / "vendor" / "comfyui_custom_nodes"
        for folder, symbols in required.items():
            with self.subTest(folder=folder):
                package = vendor_root / folder
                self.assertTrue(package.is_dir(), f"Missing bundled custom node package: {package}")
                self.assertTrue((package / "LICENSE").exists(), f"Missing bundled custom node license: {package}")
                texts = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in package.rglob("*.py"))
                for symbol in symbols:
                    self.assertIn(symbol, texts)

    def test_outpaint_prompt_bypasses_unbundled_kj_padding_node(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))

        outpaint_video.bypass_optional_preview_nodes(workflow)
        outpaint_video.bypass_demo_padding_node(workflow)
        prompt = comfy_api.workflow_to_prompt(workflow, "5076")

        class_types = {node["class_type"] for node in prompt.values()}
        self.assertNotIn("ImagePadKJ", class_types)
        self.assertIn("VHS_LoadVideo", class_types)
        self.assertIn("VHS_VideoCombine", class_types)

    def test_outpaint_prompt_sent_to_ic_lora_guide_is_global_outpaint_only(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args(
            [
                "--source",
                "input/example.mp4",
                "--comfy-dir",
                str(app.ROOT),
                "--prompt",
                "outpaint",
                "--dry-run",
            ]
        )

        with (
            mock.patch.object(outpaint_video, "copy_to_comfy_input", return_value="arp_outpaint/prepared.mp4"),
            mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint/reference.png"),
            mock.patch.object(outpaint_video, "probe_video", return_value={"width": 1280, "height": 704, "frames": 24, "fps": 24.0}),
        ):
            prompt = outpaint_video.patch_workflow(
                args,
                workflow,
                app.ROOT / "prepared.mp4",
                app.ROOT,
                "arp_outpaint/test",
                outpaint_video.combine_prompt(args.prompt, ""),
                args.negative_prompt,
                42,
            )

        self.assertEqual(prompt["2483"]["inputs"]["text"], "outpaint")
        self.assertEqual(prompt["5012"]["inputs"]["positive"], ["1241", 0])
        self.assertEqual(prompt["1241"]["inputs"]["positive"], ["2483", 0])

    def test_outpaint_command_forces_global_prompt_to_outpaint(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "chunk_seconds": "20",
                "overlap_frames": "8",
                "prompt": "stale verbose prompt that should not reach the LoRA",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )

        command = app.APP.command_for("outpaint")

        self.assertEqual(command[command.index("--prompt") + 1], "outpaint")

    def test_outpaint_manifest_sync_clears_chunk_prompt_suffixes(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            outpaint_video.write_chunk_manifest(
                manifest,
                [
                    {
                        "chunk_index": "0",
                        "start_frame": "0",
                        "end_frame": "10",
                        "seed": "42",
                        "prompt_suffix": "stale per-chunk direction",
                    }
                ],
            )

            rows = outpaint_video.sync_chunk_manifest(manifest, [(0, 0, 10)], 24.0, folder, 42)

            self.assertEqual(rows[0]["prompt_suffix"], "")

    def test_colormnet_correlation_extension_install_is_opt_in(self) -> None:
        downloader_path = app.ROOT / "vendor" / "comfyui_custom_nodes" / "reference-video-colorization" / "colormnet" / "downloader.py"
        spec = importlib.util.spec_from_file_location("colormnet_downloader_under_test", downloader_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(module.optional_correlation_install_enabled())

        for value in ("1", "true", "yes", "on"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"COLORMNET_INSTALL_CORRELATION_EXTENSION": value}, clear=True):
                self.assertTrue(module.optional_correlation_install_enabled())

    def test_single_reference_rejects_missing_source_before_qwen_startup(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            missing = Path(tmp_text) / "missing.png"
            output = Path(tmp_text) / "output.png"
            argv = [
                "generate_single_reference.py",
                "--source-image",
                str(missing),
                "--output",
                str(output),
                "--workflow",
                str(app.ROOT / "workflows" / "qwen_image_edit" / "Image Edit (Qwen 2511).json"),
                "--dry-run",
            ]

            with mock.patch.object(sys, "argv", argv), mock.patch.object(generate_single_reference.qwen, "main_with_args") as qwen_main:
                with self.assertRaisesRegex(FileNotFoundError, "Reference source image not found"):
                    generate_single_reference.main()

            qwen_main.assert_not_called()

    def test_bundled_qwen_2511_subgraph_patches_to_gguf_by_default(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.png"
            source.write_bytes(b"placeholder")
            comfy_dir = folder / "comfy"
            (comfy_dir / "input").mkdir(parents=True)
            workflow = qwen_colorize_references.load_workflow(app.ROOT / "workflows" / "qwen_image_edit" / "Image Edit (Qwen 2511).json")
            args = argparse.Namespace(comfy_dir=comfy_dir, model_backend="gguf", gguf_model="qwen-image-edit-2511-Q4_K_M.gguf")

            qwen_colorize_references.patch_qwen_model_backend(args, workflow)
            prompt = qwen_colorize_references.patch_workflow(args, workflow, source, folder / "output.png", "Colorize this image.")

            self.assertEqual(prompt["169"]["inputs"]["seed"], 1)
            self.assertEqual(prompt["169"]["inputs"]["control_after_generate"], "fixed")
            self.assertEqual(prompt["161"]["class_type"], "UnetLoaderGGUF")
            self.assertEqual(prompt["161"]["inputs"]["unet_name"], "qwen-image-edit-2511-Q4_K_M.gguf")

    def test_qwen_completion_copies_produced_image_to_requested_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.png"
            output = folder / "output.png"
            produced = folder / "comfy-output.png"
            workflow = folder / "workflow.json"
            manifest = folder / "manifest.csv"
            comfy_dir = folder / "comfy"
            comfy_output = folder / "comfy-output"
            source.write_bytes(b"source image bytes")
            produced.write_bytes(b"produced image bytes")
            workflow.write_text("{}", encoding="utf-8")
            comfy_dir.mkdir()
            comfy_output.mkdir()
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_reference", "color_reference"])
                writer.writeheader()
                writer.writerow({"source_reference": str(source), "color_reference": str(output)})

            args = qwen_colorize_references.build_parser().parse_args(
                [
                    "--manifest",
                    str(manifest),
                    "--workflow",
                    str(workflow),
                    "--comfy-dir",
                    str(comfy_dir),
                    "--comfy-output-root",
                    str(comfy_output),
                    "--model-backend",
                    "safetensors",
                    "--no-normalize-to-source-size",
                    "--force",
                ]
            )

            with (
                mock.patch.object(qwen_colorize_references, "ensure_qwen_image_edit_models"),
                mock.patch.object(qwen_colorize_references, "wait_for_comfy"),
                mock.patch.object(qwen_colorize_references, "patch_qwen_model_backend"),
                mock.patch.object(qwen_colorize_references, "patch_workflow", return_value={}),
                mock.patch.object(qwen_colorize_references, "queue_prompt", return_value="prompt-id"),
                mock.patch.object(qwen_colorize_references, "wait_for_prompt", return_value={}),
                mock.patch.object(qwen_colorize_references, "extract_output_files", return_value=[produced]),
                mock.patch.object(qwen_colorize_references, "newest_output", return_value=produced),
            ):
                self.assertEqual(qwen_colorize_references.main_with_args(args), 0)

            self.assertEqual(output.read_bytes(), b"produced image bytes")

    def test_outpaint_chunk_rows_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "chunks.csv"
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "10",
                    "start_seconds": "0.000000",
                    "end_seconds": "0.416667",
                    "seed": "42",
                    "prompt_suffix": "",
                    "prepared_path": "prepared.mp4",
                    "raw_path": "raw.mp4",
                }
            ]

            app.write_outpaint_chunk_rows(manifest, rows)

            self.assertEqual(app.read_outpaint_chunk_rows(manifest)[0]["raw_path"], "raw.mp4")

    def test_outpaint_chunk_rows_do_not_rewrite_identical_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "chunks.csv"
            rows = [{"chunk_index": "0", "start_frame": "0", "end_frame": "10", "seed": "42"}]

            app.write_outpaint_chunk_rows(manifest, rows)
            first_mtime = manifest.stat().st_mtime_ns
            app.write_outpaint_chunk_rows(manifest, rows)

            self.assertEqual(manifest.stat().st_mtime_ns, first_mtime)

    def test_outpaint_chunk_save_can_clear_custom_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "chunks.csv"
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "120",
                    "start_seconds": "0.000000",
                    "end_seconds": "5.000000",
                    "custom_seconds": "5.000",
                    "seed": "42",
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)

            with mock.patch.object(server, "outpaint_chunks_state", return_value={"manifest": str(manifest)}):
                app.update_outpaint_chunk(
                    0,
                    seed="43",
                    prompt_suffix="",
                    custom_seconds="5.000",
                    custom_length=False,
                )

            stored = app.read_outpaint_chunk_rows(manifest)[0]
            self.assertEqual(stored["seed"], "43")
            self.assertEqual(stored["custom_seconds"], "")

    def test_clearing_outpaint_guide_deletes_cached_chunk_guides(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "demo_chunks.csv"
            guide_dir = app.ROOT / "intermediate" / "outpaint_anchors" / manifest.stem
            guide_dir.mkdir(parents=True, exist_ok=True)
            current = guide_dir / "chunk_0000_guide_qwen.png"
            older = guide_dir / "chunk_0000_middle_qwen.png"
            other = guide_dir / "chunk_0001_guide_qwen.png"
            for path in (current, older, other, current.with_suffix(".png.sig.json")):
                path.write_bytes(b"cached")
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "10",
                    "start_seconds": "0",
                    "end_seconds": "1",
                    "seed": "42",
                    "anchor_image": app.rel(current),
                    "anchor_position": "guide",
                    "anchor_seconds": "0.5",
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)

            try:
                with mock.patch.object(server, "outpaint_chunks_state", return_value={"manifest": str(manifest)}):
                    app.clear_outpaint_anchor(0)

                self.assertFalse(current.exists())
                self.assertFalse(older.exists())
                self.assertFalse(current.with_suffix(".png.sig.json").exists())
                self.assertTrue(other.exists())
                self.assertEqual(app.read_outpaint_chunk_rows(manifest)[0]["anchor_image"], "")
            finally:
                shutil.rmtree(guide_dir, ignore_errors=True)

    def test_reference_manifest_read_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "refs.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                handle.write("# source_video=input/example.mp4\n")
                writer = csv.DictWriter(handle, fieldnames=["enabled", "end", "source_reference", "color_reference", "prompt"])
                writer.writeheader()
                writer.writerow(
                    {
                        "enabled": "true",
                        "end": "00:00:01.000",
                        "source_reference": "bw.png",
                        "color_reference": "color.png",
                        "prompt": "",
                    }
                )

            source, fields, rows = app.read_manifest_details(manifest)

            self.assertEqual(source, "input/example.mp4")
            self.assertIn("color_reference", fields)
            self.assertEqual(rows[0]["source_reference"], "bw.png")

    def test_shot_fade_marker_round_trips_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "refs.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["enabled", "end", "source_reference", "color_reference", "prompt"])
                writer.writeheader()
                writer.writerow({"enabled": "true", "end": "00:00:01.000", "source_reference": "a.png", "color_reference": "a_color.png", "prompt": ""})
                writer.writerow({"enabled": "true", "end": "00:00:02.000", "source_reference": "b.png", "color_reference": "b_color.png", "prompt": ""})

            app.update_shot_fade(str(manifest), 0, True, "0.5")
            _source, fields, rows = app.read_manifest_details(manifest)

            self.assertIn("fade_to_next", fields)
            self.assertEqual(rows[0]["fade_to_next"], "true")
            self.assertEqual(rows[0]["crossfade_seconds"], "0.5")

    def test_colorize_plan_extends_fading_transition_chunks(self) -> None:
        rows = [
            {"end": "00:00:01.000", "fade_to_next": "true", "crossfade_seconds": "0.5"},
            {"end": "00:00:02.000"},
        ]

        plan, transitions = colorize_video.shot_plan(rows, total_frames=48, fps=24.0)

        self.assertEqual(transitions[0], 12)
        self.assertEqual(plan[0]["start"], 0)
        self.assertEqual(plan[0]["end"], 30)
        self.assertEqual(plan[1]["start"], 18)
        self.assertEqual(plan[1]["end"], 48)

    def test_outpaint_overlap_context_stops_before_guide_inside_overlap(self) -> None:
        self.assertEqual(outpaint_video.overlap_context_before_anchor(8, "0.125", 24.0, 100), 3)
        self.assertEqual(outpaint_video.overlap_context_before_anchor(8, "1.0", 24.0, 100), 8)
        self.assertEqual(outpaint_video.overlap_context_before_anchor(8, "0", 24.0, 100), 0)

    def test_outpaint_overlap_context_rejects_mixed_geometry(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            chunk = folder / "chunk.mp4"
            previous = folder / "previous.mp4"

            with (
                mock.patch.object(outpaint_video, "probe_video") as probe,
                mock.patch.object(outpaint_video.subprocess, "run") as run,
                mock.patch.object(outpaint_video, "replace_with_retry") as replace,
            ):
                probe.side_effect = [
                    {"width": 1280, "height": 720, "frames": 8},
                    {"width": 1280, "height": 704, "frames": 8},
                ]
                previous.write_bytes(b"placeholder")

                with self.assertRaisesRegex(RuntimeError, "working canvas should stay"):
                    outpaint_video.inject_overlap_context("ffmpeg", chunk, previous, 3, 24.0, False)

            run.assert_not_called()
            replace.assert_not_called()

    def test_outpaint_overlap_context_skips_first_new_frame_on_matching_geometry(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            chunk = folder / "chunk.mp4"
            previous = folder / "previous.mp4"

            with (
                mock.patch.object(outpaint_video, "probe_video") as probe,
                mock.patch.object(outpaint_video.subprocess, "run") as run,
                mock.patch.object(outpaint_video, "replace_with_retry") as replace,
            ):
                probe.side_effect = [
                    {"width": 1280, "height": 704, "frames": 8},
                    {"width": 1280, "height": 704, "frames": 8},
                ]
                previous.write_bytes(b"placeholder")

                result = outpaint_video.inject_overlap_context("ffmpeg", chunk, previous, 3, 24.0, False)

            self.assertNotEqual(result, chunk)
            filter_text = run.call_args.args[0][7]
            self.assertNotIn("scale=1280:720", filter_text)
            self.assertIn("trim=start_frame=4:end_frame=8", filter_text)
            replace.assert_called_once()

    def test_command_construction_for_outpaint_uses_overview_source(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "chunk_seconds": "20",
                "overlap_frames": "8",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )

        command = app.APP.command_for("outpaint")

        self.assertIn("--source", command)
        self.assertIn("input/example.mp4", command)
        self.assertIn("--chunk-manifest", command)

    def test_outpaint_stage_defaults_to_longer_chunks(self) -> None:
        outpaint_stage = next(stage for stage in app.STAGES if stage.key == "outpaint")
        chunk_field = next(field for field in outpaint_stage.fields if field[0] == "chunk_seconds")

        self.assertEqual(chunk_field[3], "20")

    def test_comfy_node_check_reports_missing_custom_nodes(self) -> None:
        with mock.patch.object(comfy_api, "object_info", return_value={"UnetLoaderGGUF": {}}):
            with self.assertRaisesRegex(RuntimeError, "ComfyUI-LTXVideo"):
                comfy_api.ensure_node_types(
                    "http://127.0.0.1:8188",
                    {"LTXVImgToVideoConditionOnly": "ComfyUI-LTXVideo", "UnetLoaderGGUF": "ComfyUI-GGUF"},
                    "outpainting workflow",
                )

    def test_source_section_names_include_trim_points(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "12", "section_end": "24"})

        first = app.source_section_output_for(app.APP.settings)
        app.APP.settings["global"].update({"section_start": "45", "section_end": "60"})
        second = app.source_section_output_for(app.APP.settings)

        self.assertNotEqual(first, second)
        self.assertIn("0000012000_0000024000", first.name)
        self.assertIn("0000045000_0000060000", second.name)

    def test_pipeline_source_uses_section_when_trim_points_are_set(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "12", "section_end": "24"})

        self.assertIn("source_sections", app.pipeline_source_text(app.APP.settings))

    def test_project_payload_round_trips_settings_with_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "demo.arpp"
            app.APP.settings["global"].update({"source": "input/example.mp4"})
            path.write_text(json.dumps(app.project_payload(app.APP.settings)), encoding="utf-8")

            loaded = app.read_project_file(path)

        self.assertEqual(loaded["global"]["source"], "input/example.mp4")
        self.assertIn("schema_version", app.project_payload(app.APP.settings))

    def test_project_bundle_includes_reference_assets_without_openai_key(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "refs.csv"
            source = folder / "bw.png"
            color = folder / "color.png"
            project = folder / "demo.arpp"
            source.write_bytes(b"bw")
            color.write_bytes(b"color")
            app.write_manifest_details(
                manifest,
                "input/example.mp4",
                ["enabled", "end", "source_reference", "color_reference"],
                [{"enabled": "true", "end": "00:00:01.000", "source_reference": app.rel(source), "color_reference": app.rel(color)}],
            )
            settings = copy.deepcopy(app.APP.settings)
            settings["references"].update({"manifest": app.rel(manifest), "openai_api_key": "sk-secret"})

            app.write_project_file(project, settings)

            with zipfile.ZipFile(project) as archive:
                names = set(archive.namelist())
                payload = json.loads(archive.read("project.json").decode("utf-8"))

            self.assertIn(app.rel(manifest).replace("\\", "/"), names)
            self.assertIn(app.rel(source).replace("\\", "/"), names)
            self.assertIn(app.rel(color).replace("\\", "/"), names)
            self.assertNotIn("openai_api_key", payload["settings"]["references"])

    def test_openai_reference_command_requires_key_and_uses_selected_model(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "refs.csv"
            source = folder / "bw.png"
            color = folder / "color.png"
            source.write_bytes(b"bw")
            app.write_manifest_details(
                manifest,
                "input/example.mp4",
                ["enabled", "end", "source_reference", "color_reference", "prompt"],
                [{"enabled": "true", "end": "00:00:01.000", "source_reference": app.rel(source), "color_reference": app.rel(color), "prompt": "warm highlights"}],
            )
            app.APP.settings["references"].update({"openai_api_key": "", "openai_image_model": "gpt-image-2"})

            with self.assertRaisesRegex(RuntimeError, "OpenAI API key"):
                app.openai_reference_regeneration_command(str(manifest), 0)

            app.APP.settings["references"].update({"openai_api_key": "sk-test", "openai_image_model": "gpt-image-1"})
            command, output = app.openai_reference_regeneration_command(str(manifest), 0)

            self.assertIn("openai_generate_reference.py", " ".join(command))
            self.assertIn("--manifest", command)
            self.assertIn("--row-index", command)
            self.assertEqual(command[command.index("--model") + 1], "gpt-image-1")
            self.assertEqual(output, app.rel(color))

    def test_openai_manifest_command_can_send_nearby_reference_images(self) -> None:
        app.APP.settings["references"].update(
            {
                "manifest": "manifests/references/demo.csv",
                "method": "openai",
                "openai_api_key": "sk-test",
                "openai_image_model": "gpt-image-2",
                "openai_send_references": "true",
            }
        )

        command = app.APP.command_for("references")

        self.assertIn("openai_generate_reference.py", " ".join(command))
        self.assertIn("--reference-count", command)
        self.assertNotIn("qwen_colorize_references.py", " ".join(command))

    def test_openai_nearby_references_prefer_previous_then_later(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            refs = []
            rows = []
            for index in range(5):
                color = folder / f"color_{index}.png"
                if index in {0, 2, 3, 4}:
                    color.write_bytes(b"ref")
                refs.append(color)
                rows.append({"color_reference": app.rel(color)})

            chosen = openai_generate_reference.nearby_reference_images(rows, 1, 3)
            early = openai_generate_reference.nearby_reference_images(rows, 0, 3)

        self.assertEqual(chosen, [refs[0], refs[2], refs[3]])
        self.assertEqual(early, [refs[2], refs[3], refs[4]])

    def test_project_save_suggestion_uses_last_browse_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            app.APP.settings["global"].update({"source": "input/example.mp4", "last_browse_dir": str(folder)})

            suggestion = app.project_save_suggestion(app.APP.settings)

        self.assertEqual(suggestion.parent, folder)
        self.assertEqual(suggestion.name, "example.arpp")

    def test_browse_initial_path_uses_last_browse_dir_without_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            app.APP.settings["global"]["last_browse_dir"] = str(folder)

            self.assertEqual(app.browse_initial_path("project_open", ""), folder)

    def test_browse_initial_path_prefers_last_dir_over_existing_current_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text, tempfile.TemporaryDirectory() as old_text:
            remembered = Path(tmp_text)
            old = Path(old_text)
            current = old / "layer.mp4"
            current.write_bytes(b"placeholder")
            app.APP.settings["global"]["last_browse_dir"] = str(remembered)

            self.assertEqual(app.browse_initial_path("save", str(current)), remembered / "layer.mp4")
            self.assertEqual(app.browse_initial_path("file", str(current)), remembered)

    def test_colorized_outputs_include_both_methods(self) -> None:
        outputs = app.colorized_outputs_for_manifest("manifests/references/colorize_manifest_demo_shots_auto.csv", "both")

        self.assertEqual(len(outputs), 2)
        self.assertTrue(outputs[0].endswith("_deepexemplar_colorized.mp4"))
        self.assertTrue(outputs[1].endswith("_colormnet_colorized.mp4"))

    def test_colorization_command_can_request_both_methods(self) -> None:
        app.APP.settings["colour"].update({"manifest": "manifests/references/colorize_manifest_demo_shots_auto.csv", "method": "both"})

        command = app.APP.command_for("colour")

        self.assertIn("--method", command)
        self.assertIn("both", command)
        self.assertNotIn("--output", command)

    def test_skip_outpainting_uses_pipeline_source_for_shot_detection(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "true", "section_start": "0", "section_end": ""})

        app.APP.hydrate_stage_inputs("global")
        stage_keys = [stage.key for stage in app.APP.active_stages()]
        command = app.APP.command_for("shots")

        self.assertNotIn("outpaint", stage_keys)
        self.assertIn("shots", stage_keys)
        self.assertEqual(app.APP.settings["shots"]["outpainted_video"], "input/example.mp4")
        self.assertIn("input/example.mp4", command)

    def test_new_outpaint_source_does_not_hydrate_empty_manifest_as_repo_root(self) -> None:
        app.APP.settings["global"].update({"source": "input/new-source.mp4", "expand_outpaint": "true", "colorize": "true", "section_start": "0", "section_end": ""})
        app.APP.settings.setdefault("outpaint", {}).update({"target_aspect": "16:9", "target_height": "480"})

        with mock.patch.object(server, "newest", return_value=None):
            app.APP.hydrate_stage_inputs("global")

        self.assertEqual(app.APP.settings["shots"]["outpainted_video"], "")
        self.assertEqual(app.APP.settings["references"]["manifest"], "")
        self.assertEqual(app.APP.settings["colour"]["manifest"], "")
        self.assertEqual(app.APP.settings["recomp"]["manifest"], "")
        self.assertEqual(app.APP.expected_outputs("shots"), [])

    def test_outpaint_progress_surfaces_active_comfy_chunk_globally(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "outpaint"
        app.APP.running_stage = "Outpainting"
        app.APP.run_started_at = time.time() - 120
        app.APP.log = [
            "Outpaint chunk 1/3: frames 0-480",
            "Wrote raw Comfy chunk 1: chunk_0000.mp4",
            "Outpaint chunk 2/3: frames 472-952",
            "Sending prompt nodes: {'5076': 'VHS_VideoCombine'}",
            "Queued ComfyUI prompt: prompt-id",
        ]

        try:
            stage_progress = app.APP.estimate_running_progress()
            global_progress = app.APP.phase_progress()["global"]
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertIn("Chunk 2/3 rendering in ComfyUI (1 done), ETA", stage_progress["label"])
        self.assertIn("Chunk 2/3 rendering in ComfyUI", global_progress["label"])

    def test_reference_progress_ignores_previous_stage_writes(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "references"
        app.APP.running_stage = "Reference Generation"
        app.APP.run_started_at = time.time() - 60
        app.APP.log = [
            "Wrote source frame 0000: source.png",
            "Wrote source frame 0001: source.png",
            "Wrote manifest: refs.csv",
            r"> python scripts\qwen_colorize_references.py --manifest refs.csv",
            "Rows: 11",
            "Colorize 0000: source.png -> color.png",
            "Queued ComfyUI prompt: prompt-id",
            "Wrote color.png",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertEqual(progress["label"], "1/11 references")

    def test_colour_progress_ignores_previous_process_finished_lines(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "colour"
        app.APP.running_stage = "Colorization"
        app.APP.run_started_at = time.time() - 60
        app.APP.log = [
            r"> python scripts\qwen_colorize_references.py --manifest refs.csv",
            "Wrote reference.png",
            "Process finished with exit code 0.",
            r"> python scripts\colorize_video.py --manifest refs.csv --method both",
            "Colorize segment 11/11 with deepexemplar: frames 1343-1440 using ref.png",
            "Wrote colorized video: deepexemplar.mp4",
            "Colorize segment 4/11 with colormnet: frames 527-632 using ref.png",
            "Sending prompt nodes: {'1': 'VHS_LoadVideo'}",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertEqual(progress["label"], "Colorizing segment 4/11")

    def test_blank_project_defaults_outpainting_visible(self) -> None:
        with mock.patch.object(app, "SETTINGS_FILE", Path("missing-settings.json")), mock.patch.object(app, "newest", return_value=None):
            settings = app.load_settings()

        self.assertEqual(settings["global"]["expand_outpaint"], "true")

    def test_blank_loaded_project_defaults_outpainting_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "blank.arpp"
            payload = app.project_payload(app.APP.settings)
            payload["settings"] = {"global": {"source": "", "expand_outpaint": "false", "colorize": "true"}}
            path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = app.read_project_file(path)

        self.assertEqual(loaded["global"]["expand_outpaint"], "true")

    def test_section_preview_times_are_relative_to_trim_start(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "12", "section_end": "24"})

        self.assertAlmostEqual(app.section_relative_seconds(app.APP.settings, 12), 0.0)
        self.assertAlmostEqual(app.section_relative_seconds(app.APP.settings, 18.5), 6.5)
        self.assertAlmostEqual(app.section_relative_seconds(app.APP.settings, 30), 12.0)

    def test_outpaint_chunks_prepares_section_before_reading_it(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "12", "section_end": "24"})

        with mock.patch.object(server, "ensure_source_section_clip") as ensure, mock.patch.object(server, "resolve_video_source") as resolve_source:
            resolve_source.return_value = Path("missing-section.mp4")
            state = app.outpaint_chunks_state(app.APP.settings)

        ensure.assert_called_once_with(app.APP.settings)
        self.assertIn("not a readable file", state["error"])

    def test_media_clip_rejects_missing_source(self) -> None:
        with self.assertRaises(FileNotFoundError):
            app.media_clip_path(app.ROOT / "does-not-exist.mp4", 0, 1, "smoke")

    def test_frame_preview_reuses_fresh_existing_thumbnail(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source.write_bytes(b"video placeholder")
            target_dir = folder / "previews"
            target_dir.mkdir()
            target = target_dir / f"{app.safe_preview_name(source)}_thumb.jpg"
            target.write_bytes(b"cached thumbnail")

            with mock.patch.object(server, "local_tool", return_value="ffmpeg"), mock.patch.object(server.subprocess, "run") as run:
                self.assertEqual(app.extract_video_frame_at(source, target_dir, "thumb", 0), app.rel(target))

            run.assert_not_called()

    def test_files_for_skips_files_deleted_during_refresh(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            rel_folder = app.rel(folder)
            disappearing = folder / "vanishing.txt"
            disappearing.write_text("briefly here", encoding="utf-8")
            stage = app.Stage("smoke", "Smoke", "", (rel_folder,), (), ())
            real_stat = Path.stat
            calls = {"target": 0}

            def stat_once_then_missing(path: Path, *args, **kwargs):
                if path == disappearing:
                    calls["target"] += 1
                    if calls["target"] >= 2:
                        raise FileNotFoundError(str(path))
                return real_stat(path, *args, **kwargs)

            with mock.patch.object(Path, "stat", stat_once_then_missing):
                self.assertEqual(app.APP.files_for(stage), [])

    def test_state_endpoint_returns_json(self) -> None:
        server = app.create_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/state", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        finally:
            server.shutdown()
            server.server_close()

        self.assertIn("stages", payload)
        self.assertIn("settings", payload)

    def test_root_serves_static_frontend_shell(self) -> None:
        server = app.create_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5) as response:
                html = response.read().decode("utf-8")
        finally:
            server.shutdown()
            server.server_close()

        self.assertIn('/static/styles.css', html)
        self.assertIn('/static/js/core.js', html)
        self.assertIn('/static/js/render-cache.js', html)
        self.assertIn('/static/js/app.js', html)


if __name__ == "__main__":
    unittest.main()
