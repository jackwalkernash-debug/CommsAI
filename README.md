# CommsAI

A Windows desktop prototype for real-time incoming voice translation in games.

The customer-facing app does not require SteelSeries Sonar, Python, or the .NET SDK to be installed manually. The release workflow builds:

- a self-contained Windows desktop application;
- a packaged local GPU transcription backend;
- an installer;
- a portable ZIP.

## Current MVP scope

- Captures audio directly from a selected Windows output device using WASAPI loopback.
- Segments likely speech locally.
- Sends completed speech segments to a persistent Faster-Whisper backend.
- Uses the NVIDIA GPU through CUDA.
- Displays the detected language, original transcript, and literal English translation.
- Can speak the English translation through Windows TTS.
- Stores logs and temporary recordings locally.
- Does not require SteelSeries Sonar.

This MVP captures the complete selected Windows output. CS2 game sounds and voice chat are therefore mixed together. Process-specific CS2 capture and stronger speech enhancement are later milestones.

## Build without installing development tools

1. Create a new private GitHub repository.
2. Upload the contents of this folder to the repository.
3. Open the repository's **Actions** tab.
4. Run **Build CommsAI Windows Release**.
5. When it finishes, download:
   - `CommsAI-Installer`
   - or `CommsAI-Portable`

The cloud build can be large because it packages the CUDA runtime and local speech-recognition backend.

## First launch

1. Install or extract CommsAI.
2. Start the application.
3. Select the Windows output device carrying CS2 or the test video.
4. Select `large-v3` for the accuracy benchmark.
5. Click **Start**.
6. The first model load downloads the model from Hugging Face and may take several minutes.
7. Play a team-comms video and compare:
   - Original transcript
   - Literal English translation
   - YouTube subtitles

## Requirements

- Windows 10/11 x64
- NVIDIA GPU with a recent driver
- Internet connection for first model download
- Several gigabytes of free storage

## Privacy

Audio processing is local after the model has been downloaded. Temporary WAV segments are deleted after processing unless development logging is enabled.
