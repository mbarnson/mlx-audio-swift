# PRD-03: Port LFM2.5-Audio to mlx-audio-swift

See full PRD details in the Claude Code prompt below.
This file is a reference marker - the full instructions were passed to Claude Code directly.

Key points:
- Port Python mlx-audio LFM2.5-Audio to native Swift in MLXAudioSTS module
- Reference Python code at /Users/patbarnson/devel/mlx-audio-python/mlx_audio/sts/models/lfm_audio/
- Reference LFM2 backbone at mlx-lm (pip installed or GitHub)
- Work in the existing MLXAudioSTS placeholder module
- 10 phases: Setup, Config, Backbone, Conformer, Depthformer, Model, Detokenizer, Processor, API, Testing
