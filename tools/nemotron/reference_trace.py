#!/usr/bin/env python3
"""Generate deterministic Nemotron 3.5 streaming reference tensors.

This harness intentionally uses the Hugging Face/NVIDIA model implementation rather
than Voxline native code.  It emits contiguous tensors for the first two streaming
chunks plus the complete RNNT token sequence so Android/native backends can be
compared against an independent implementation.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import AutoProcessor, Nemotron3_5AsrForRNNT
from transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr import (
    Nemotron3_5AsrRNNTDecoderCache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local snapshot directory")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--right-context", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path, help="Output prefix without extension")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def tensor_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().contiguous().numpy()


def snapshot_attention(cache: Any) -> tuple[np.ndarray, np.ndarray]:
    keys = torch.stack([layer.keys.detach().cpu() for layer in cache.layers], dim=0)
    values = torch.stack([layer.values.detach().cpu() for layer in cache.layers], dim=0)
    return tensor_np(keys), tensor_np(values)


def snapshot_padding(cache: Any, prefix: str, arrays: dict[str, np.ndarray]) -> list[str]:
    names: list[str] = []
    for layer_name, layer in sorted(cache.layers.items()):
        key = f"{prefix}_padding_{layer_name.replace('.', '_')}"
        arrays[key] = tensor_np(layer.cache)
        names.append(key)
    return names


def prompt_fusion(model: Nemotron3_5AsrForRNNT, hidden: torch.Tensor, prompt_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    one_hot = torch.nn.functional.one_hot(
        prompt_ids,
        num_classes=model.config.num_prompts,
    ).to(hidden.dtype)
    expanded = one_hot[:, None, :].expand(-1, hidden.shape[1], -1)
    fused = model.prompt_projector(torch.cat([hidden, expanded], dim=-1))
    projected = model.encoder_projector(fused)
    return expanded, fused, projected


def make_feature_chunks(
    processor: Any,
    audio: np.ndarray,
    sampling_rate: int,
    language: str,
) -> tuple[Any, list[torch.Tensor]]:
    first = processor(
        audio[: processor.num_samples_first_audio_chunk],
        sampling_rate=sampling_rate,
        is_streaming=True,
        is_first_audio_chunk=True,
        language=language,
        return_tensors="pt",
    )
    chunks = [first.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]]
    mel_frame_idx = processor.num_mel_frames_first_audio_chunk
    hop_length = processor.feature_extractor.hop_length
    half_fft = processor.feature_extractor.n_fft // 2
    while True:
        start_idx = mel_frame_idx * hop_length - half_fft
        end_idx = start_idx + processor.num_samples_per_audio_chunk
        if end_idx >= audio.shape[0]:
            break
        inputs = processor(
            audio[start_idx:end_idx],
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=False,
            language=language,
            return_tensors="pt",
        )
        chunks.append(inputs.input_features)
        mel_frame_idx += processor.num_mel_frames_per_audio_chunk
    return first, chunks


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    audio, sampling_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if audio.ndim != 1:
        raise ValueError("Reference input must be mono")
    if sampling_rate != 16000:
        raise ValueError(f"Reference input must be 16 kHz, got {sampling_rate}")
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    local_only = Path(args.model).exists()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=local_only)
    processor.set_num_lookahead_tokens(args.right_context)
    model = Nemotron3_5AsrForRNNT.from_pretrained(
        args.model,
        local_files_only=local_only,
        dtype=torch.float32,
    ).eval()

    first_inputs, chunks = make_feature_chunks(processor, audio, sampling_rate, args.language)
    if len(chunks) < 2:
        raise ValueError("Audio is too short to produce two streaming chunks")

    arrays: dict[str, np.ndarray] = {
        "pcm_float32": audio,
        "feature_first": tensor_np(chunks[0]),
        "feature_second": tensor_np(chunks[1]),
        "prompt_ids": first_inputs.prompt_ids.detach().cpu().numpy(),
    }
    timings_ms: dict[str, float] = {}

    with torch.inference_mode():
        start = time.perf_counter()
        encoder_first = model.encoder(
            input_features=chunks[0],
            num_lookahead_tokens=args.right_context,
            use_cache=True,
            output_attention_mask=False,
        )
        timings_ms["encoder_first"] = (time.perf_counter() - start) * 1000.0

        arrays["encoder_first_hidden"] = tensor_np(encoder_first.last_hidden_state)
        arrays["attention_first_keys"], arrays["attention_first_values"] = snapshot_attention(
            encoder_first.past_key_values
        )
        first_padding_names = snapshot_padding(encoder_first.padding_cache, "first", arrays)
        arrays["prompt_first_one_hot"], prompt_first_fused, prompt_first_projected = (
            tensor_np(value)
            for value in prompt_fusion(model, encoder_first.last_hidden_state, first_inputs.prompt_ids)
        )
        arrays["prompt_first_fused"] = prompt_first_fused
        arrays["encoder_first_projected"] = prompt_first_projected

        # Snapshot caches before the second call because Transformers updates cache
        # objects in place.
        past_key_values = encoder_first.past_key_values
        padding_cache = encoder_first.padding_cache
        start = time.perf_counter()
        encoder_second = model.encoder(
            input_features=chunks[1],
            past_key_values=past_key_values,
            padding_cache=padding_cache,
            num_lookahead_tokens=args.right_context,
            use_cache=True,
            output_attention_mask=False,
        )
        timings_ms["encoder_second"] = (time.perf_counter() - start) * 1000.0

        arrays["encoder_second_hidden"] = tensor_np(encoder_second.last_hidden_state)
        arrays["attention_second_keys"], arrays["attention_second_values"] = snapshot_attention(
            encoder_second.past_key_values
        )
        second_padding_names = snapshot_padding(encoder_second.padding_cache, "second", arrays)
        prompt_second_one_hot, prompt_second_fused, prompt_second_projected = prompt_fusion(
            model, encoder_second.last_hidden_state, first_inputs.prompt_ids
        )
        arrays["prompt_second_one_hot"] = tensor_np(prompt_second_one_hot)
        arrays["prompt_second_fused"] = tensor_np(prompt_second_fused)
        arrays["encoder_second_projected"] = tensor_np(prompt_second_projected)

        blank_id = int(model.config.blank_token_id)
        decoder_cache = Nemotron3_5AsrRNNTDecoderCache(model.config)
        decoder_input = torch.tensor([[blank_id]], dtype=torch.long)
        start = time.perf_counter()
        predictor_output = model.decoder(decoder_input, cache=decoder_cache)
        timings_ms["predictor"] = (time.perf_counter() - start) * 1000.0
        arrays["predictor_output"] = tensor_np(predictor_output)
        arrays["predictor_cache"] = tensor_np(decoder_cache.cache)
        arrays["predictor_hidden_state"] = tensor_np(decoder_cache.hidden_state)
        arrays["predictor_cell_state"] = tensor_np(decoder_cache.cell_state)

        projected_first = torch.from_numpy(arrays["encoder_first_projected"])
        start = time.perf_counter()
        joint_logits = model.joint(
            encoder_hidden_states=projected_first[:, :1, None, :],
            decoder_hidden_states=predictor_output[:, None, :, :],
        ).squeeze(2)
        timings_ms["joint"] = (time.perf_counter() - start) * 1000.0
        arrays["joint_logits_first_frame"] = tensor_np(joint_logits)
        top = torch.topk(joint_logits[0, 0], min(args.top_n, joint_logits.shape[-1]))
        arrays["joint_top_ids"] = top.indices.detach().cpu().numpy()
        arrays["joint_top_logits"] = tensor_np(top.values)

        def chunk_generator():
            for chunk in chunks:
                yield chunk

        start = time.perf_counter()
        generation_inputs = {**first_inputs, "input_features": chunk_generator()}
        generated = model.generate(**generation_inputs)
        timings_ms["generate"] = (time.perf_counter() - start) * 1000.0

    sequence = generated.sequences[0].detach().cpu().tolist()
    emitted = [token for token in sequence if token != blank_id]
    transcript = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
    arrays["generated_sequence"] = np.asarray(sequence, dtype=np.int64)
    arrays["emitted_token_ids"] = np.asarray(emitted, dtype=np.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    pcm16 = np.clip(np.rint(audio * 32768.0), -32768, 32767).astype("<i2")
    pcm16.tofile(args.output.with_suffix(".s16le"))

    metadata = {
        "schema": 1,
        "model": args.model,
        "audio": str(args.audio),
        "language": args.language,
        "right_context": args.right_context,
        "streaming_latency_ms": processor.streaming_latency_ms,
        "sampling_rate": sampling_rate,
        "audio_samples": int(audio.shape[0]),
        "blank_id": blank_id,
        "prompt_ids": first_inputs.prompt_ids.detach().cpu().tolist(),
        "feature_chunk_shapes": [list(chunk.shape) for chunk in chunks],
        "encoder_first_shape": list(arrays["encoder_first_hidden"].shape),
        "encoder_second_shape": list(arrays["encoder_second_hidden"].shape),
        "attention_first_shape": list(arrays["attention_first_keys"].shape),
        "attention_second_shape": list(arrays["attention_second_keys"].shape),
        "padding_first_arrays": first_padding_names,
        "padding_second_arrays": second_padding_names,
        "joint_top_ids": arrays["joint_top_ids"].tolist(),
        "emitted_token_ids": emitted,
        "transcript": transcript,
        "timings_ms": timings_ms,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
