# Open-Weights Model Survey — 2026-07-31

*Written: July 31, 2026. Source links checked September 30, 2026.*

I am comparing open-weight candidates for OCR, text-to-speech, speech recognition, and translation. These tasks need different evaluations, so there is no useful single ranking across them. My selection also depends on whether the available runtime supports the local hardware. A strong CUDA result is not evidence of a working ROCm path.

The shortlist below preserves the July scope. Live sources can verify model properties without reproducing the exact July leaderboard; where the dated figure cannot be recovered, I treat it as a recorded comparison requiring a source rather than a current ranking. Community sentiment was not established in the original survey and contributes no evidence here.

## OCR

[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) is my first local candidate because it is compact and has an available [ggml-org GGUF implementation](https://huggingface.co/ggml-org/GLM-OCR-GGUF/tree/main). Its complete document pipeline includes layout processing, so a pipeline score should not automatically be assigned to bare model inference.

[Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) is worth testing for multi-page input. Baidu reports 93.92 on OmniDocBench v1.6 and demonstrations around 40 pages. That is evidence of the intended workload, not a guarantee of completeness on an arbitrary long document. The llama.cpp R-SWA implementation was tracked in [PR #24975](https://github.com/ggml-org/llama.cpp/pull/24975); support in that branch should not be confused with merged mainline behavior.

The [MinerU2.5-Pro paper](https://arxiv.org/abs/2604.04771) reports 95.69 on v1.6. It is another accuracy candidate, but page parsing and a single-pass long-document workflow are different comparisons. The original GLM-OCR figure of 95.15 on v1.6 could not be recovered in the checked primary pages; Z.ai's card instead explicitly lists 94.62 on v1.5. These values should not be mixed into a precise ordering.

The original advice to avoid every four-bit Unlimited-OCR quantization was too broad. I would compare the actual quantized path against a higher-precision reference on dense tables and long output; attention/cache settings and decoding configuration can also damage extraction.

## Text-to-speech

[Step-Audio-EditX](https://github.com/stepfun-ai/Step-Audio-EditX) is a 3B candidate for expressive speech and voice editing. Its documented path requires NVIDIA CUDA. The July notes recorded 32 GB, while the checked repository now describes a 12 GB threshold and recommends more margin. The requirement depends on the implementation revision and quantization; I have not established an official ROCm deployment.

[Kokoro 82M](https://huggingface.co/hexgrad/Kokoro-82M), with Apache-licensed weights, is the compact CPU candidate. [Chatterbox](https://github.com/resemble-ai/chatterbox), under MIT, is a voice-cloning candidate. Their actual speed and compatibility on my AMD systems remain to be measured.

The original Step-Audio Elo of 1,118 and claim that no open model occupied the commercial top ten were not recovered in a dated [Artificial Analysis arena](https://artificialanalysis.ai/text-to-speech/arena) snapshot. I cannot use them to establish a quality leader or the widest open/closed gap. Arena preference also does not directly measure cloning fidelity, language coverage, or deployment reliability.

## Speech recognition

The strongest numerical evidence in this shortlist is the common English ASR evaluation: [Granite Speech 4.1 2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) reports 5.33% mean WER and [Cohere Transcribe 2B](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) 5.42%, with Apache 2.0 releases. A 0.09-point difference in a dataset average does not establish the better model for my recordings.

Whisper large-v3-turbo is a multilingual starting candidate with a documented [whisper.cpp HIP/ROCm path](https://github.com/ggml-org/whisper.cpp#amd-rocm-gpu-support). [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) supports 52 languages and dialects; [NVIDIA Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) is a throughput candidate for its supported languages and runtime. I would choose among them by language, noise, streaming needs, timestamps, and measured local latency rather than a small aggregate WER gap.

## Translation

[Tencent's HY-MT family](https://github.com/Tencent-Hunyuan/Hy-MT) supplies 7B and 1.8B translation candidates across 33 languages, with Hy-MT2 released in May. That successor was already available by the July survey, so HY-MT1.5 should be a baseline rather than an unquestioned current default. [TranslateGemma](https://huggingface.co/google/translategemma-27b-it) offers 4B, 12B, and 27B alternatives; [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) and [SeamlessM4T](https://huggingface.co/facebook/seamless-m4t-v2-large) remain coverage candidates where their language support and license fit the task.

WMT human evaluation and vendor XCOMET results answer different questions. I would retain the language pair, test set, and evaluation method with each claim, then check terminology, meaning preservation, and document context on my own material. A vendor rank alone does not establish equivalence to a closed frontier model.

The shortlist is ready for local comparison. The clearest published comparison is English ASR under a common evaluation; OCR requires pipeline/version matching, translation requires language-pair evidence, and the historical TTS ranking remains unresolved. None of those results substitutes for measuring the complete runtime I intend to use.
