# AwakeWord

A cross-platform C# library for wake-word detection using ONNX models.

## Features

- ✅ **Real-time wake-word detection** from microphone input
- ✅ **File processing** for WAV, MP3, M4A, AIFF, and WMA files
- ✅ **Low-latency streaming** with configurable thresholds
- ✅ **ONNX-powered** ML models (openWakeWord v0.5.1)
- ✅ **Fully tested** with comprehensive unit tests

## Quick Start

### 1. Install Dependencies

The library requires:
- .NET 10.0 or later
- ONNX models (included in `models/` directory)

### 2. Real-time Detection (Microphone)

```bash
# Listen for "Hey Jarvis" (default)
dotnet run --project apps/AwakeWord.TestApp -- --models-dir models --threshold 0.5

# Listen for "Hey Mycroft"
dotnet run --project apps/AwakeWord.TestApp -- --models-dir models --wake-word mycroft --threshold 0.5
```

### 3. File Processing

```bash
# Process a WAV file (Jarvis)
dotnet run --project apps/AwakeWord.FileProcessor -- \
    --audio-file audio.wav \
    --models-dir models \
    --wake-word jarvis \
    --threshold 0.5

# Process an MP3 file (Mycroft)
dotnet run --project apps/AwakeWord.FileProcessor -- \
    --audio-file audio.mp3 \
    --models-dir models \
    --wake-word mycroft \
    --threshold 0.5
```

### 4. Programmatic Usage

```csharp
using AwakeWord.Core;

// Configure for "Hey Jarvis"
var config = new WakeWordConfig
{
    MelSpectrogramModelPath = "models/melspectrogram.onnx",
    EmbeddingModelPath = "models/embedding_model.onnx",
    WakeWordModelPath = "models/hey_jarvis_v0.1.onnx",
    WakeWord = "jarvis",
    DetectionThreshold = 0.5f
};

// Or configure for "Hey Mycroft"
var mycroftConfig = new WakeWordConfig
{
    MelSpectrogramModelPath = "models/melspectrogram.onnx",
    EmbeddingModelPath = "models/embedding_model.onnx",
    WakeWordModelPath = "models/hey_mycroft_v0.1.onnx",
    WakeWord = "mycroft",
    DetectionThreshold = 0.5f
};

using var detector = new OnnxWakeWordDetector(config);

// Load audio from file (WAV, MP3, etc.)
var samples = AudioFileLoader.LoadAudioFile("audio.mp3");

// Or process live audio chunks
short[] audioChunk = /* ... */;
if (detector.ProcessAudio(audioChunk, out var confidence))
{
    Console.WriteLine($"Wake word detected! Confidence: {confidence:0.3f}");
}
```

## Project Structure

```
AwakeWord/
├── src/
│   └── AwakeWord.Core/          # Core library
│       ├── AudioFileLoader.cs   # MP3/WAV file loading
│       ├── AudioFormat.cs       # Audio format constants
│       ├── IAwakeWordDetector.cs
│       ├── OnnxWakeWordDetector.cs  # ONNX pipeline implementation
│       └── WakeWordConfig.cs
├── apps/
│   ├── AwakeWord.TestApp/       # Microphone capture demo
│   └── AwakeWord.FileProcessor/ # File processing CLI
├── tests/
│   └── AwakeWord.Core.Tests/    # Unit tests
├── models/                       # ONNX models
│   ├── melspectrogram.onnx
│   ├── embedding_model.onnx
│   ├── hey_jarvis_v0.1.onnx
│   └── hey_mycroft_v0.1.onnx
└── docs/
    └── AUDIO_FILE_SUPPORT.md    # Audio format documentation
```

## Audio Format Support

The library supports loading audio from multiple formats:

| Format | Extension | Auto-Conversion |
|--------|-----------|-----------------|
| WAV    | .wav      | ✅ Yes          |
| MP3    | .mp3      | ✅ Yes          |
| M4A    | .m4a      | ✅ Yes          |
| AIFF   | .aiff     | ✅ Yes          |
| WMA    | .wma      | ✅ Yes          |

All audio is automatically converted to:
- **16 kHz sample rate**
- **Mono (1 channel)**
- **16-bit PCM**

See [docs/AUDIO_FILE_SUPPORT.md](docs/AUDIO_FILE_SUPPORT.md) for details.

## Running Tests

```bash
# Run all tests
dotnet test

# Run specific test class
dotnet test --filter "FullyQualifiedName~AudioActivationTests"
```

Current test coverage: **22 tests** covering:
- Configuration validation
- Audio format validation
- Detector initialization and reset
- Silence handling
- Real audio file processing
- Threshold behavior
- Streaming processing
- MP3 file support
- Multiple wake word models (Jarvis, Mycroft)

## ONNX Models

The library uses [openWakeWord](https://github.com/dscripka/openWakeWord) models:

1. **melspectrogram.onnx** - Converts raw PCM to mel-spectrogram
2. **embedding_model.onnx** - Extracts audio embeddings
3. **hey_jarvis_v0.1.onnx** - "Hey Jarvis" classifier
4. **hey_mycroft_v0.1.onnx** - "Hey Mycroft" classifier

Models are from openWakeWord v0.5.1 release.

### Supported Wake Words

| Wake Word | Model File | Usage |
|-----------|------------|-------|
| jarvis | hey_jarvis_v0.1.onnx | `--wake-word jarvis` |
| mycroft | hey_mycroft_v0.1.onnx | `--wake-word mycroft` |

## Requirements

- **.NET 10.0** or later
- **Windows** (for audio file support via Windows Media Foundation)
  - Windows 7+ or Windows Server 2008 R2+
  - For Linux/macOS, implement custom audio loader

## Performance

- **Latency**: ~80ms chunks (configurable)
- **Memory**: Minimal buffering with streaming architecture
- **CPU**: Optimized for ONNX Runtime CPU provider

## Development

### Build

```bash
dotnet build
```

### Debug in VS Code

The project includes debug configurations:
- **Launch TestApp** - Debug microphone capture app
- **Launch TestApp (Low Threshold)** - Debug with threshold 0.3
- **Debug All Tests** - Debug all unit tests
- **Debug Current Test** - Debug test at cursor

## License

Check license requirements for:
- Your application
- openWakeWord models (Apache 2.0)
- NAudio (MIT)
- Microsoft.ML.OnnxRuntime (MIT)

## Contributing

1. Follow existing code structure (one class per file)
2. Add unit tests for all new functionality
3. Update documentation as needed

## Troubleshooting

### "Models not found" error

Ensure ONNX model files are in the `models/` directory:
```bash
ls models/
# Should show: melspectrogram.onnx, embedding_model.onnx, 
# hey_jarvis_v0.1.onnx, hey_mycroft_v0.1.onnx
```

### Low detection accuracy

- Try lowering threshold: `--threshold 0.3`
- Ensure audio is clear and contains the wake word ("Hey Jarvis" or "Hey Mycroft")
- Check microphone input quality
- Verify you're using the correct model for your wake word: `--wake-word jarvis` or `--wake-word mycroft`

### Audio file format not supported

Verify the format is supported:
```csharp
AudioFileLoader.IsSupportedFormat(".mp3"); // true
AudioFileLoader.IsSupportedFormat(".flac"); // false
```

## Resources

- [openWakeWord](https://github.com/dscripka/openWakeWord) - ML models
- [NAudio](https://github.com/naudio/NAudio) - Audio I/O
- [ONNX Runtime](https://onnxruntime.ai/) - Model inference
