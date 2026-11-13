# Multimodal AI Toolkit

Local speech recognition, text to speech, CLIP and image generation pipelines with shared utilities.

## Structure

```
chattts/chat_tts_synth.py
clip_cn/task1_1N_match.py
clip_cn/task2_MN_match.py
clip_cn/task3_other_clip_compare.py
clip_cn/test_clip.py
common/clip_utils.py
common/info.py
common/init__.py
requirements.txt
scripts/check_chattts.py
scripts/check_env.py
scripts/curl_download.py
scripts/download_chattts_weights.py
scripts/download_models.py
scripts/download_test_images.py
scripts/encodec_stub.py
scripts/eval_tts_quality.py
scripts/fill_report.py
scripts/fix_chattts_deps.py
scripts/make_contact_sheet.py
scripts/make_result_table.py
scripts/make_test_audio.py
sd/task1_prompt_engineering.py
sd/task2_inference_steps.py
sd/task3_clip_eval.py
sd/test_tiny_gen_one.py
whisper/task1_realtime_record.py
whisper/task2_hf_whisper.py
whisper/whisper_test.py
```

## Setup

Install dependencies from `requirements.txt` when present:

```bash
pip install -r requirements.txt
```

Data files, model weights and generated outputs are not included; place them under the paths referenced by the configuration before running.
