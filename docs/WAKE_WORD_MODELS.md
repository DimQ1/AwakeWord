# Wake Word Models

The AwakeWord library supports multiple wake word models from the [openWakeWord](https://github.com/dscripka/openWakeWord) project.

## Supported Wake Words

| Wake Word | Model File | Command Line | Example |
|-----------|------------|--------------|---------|
| **Jarvis** | `hey_jarvis_v0.1.onnx` | `--wake-word jarvis` | "Hey Jarvis" |
| **Mycroft** | `hey_mycroft_v0.1.onnx` | `--wake-word mycroft` | "Hey Mycroft" |

## Using Different Wake Words

### Command Line (TestApp)

```bash
# Listen for "Hey Jarvis" (default)
dotnet run --project apps/AwakeWord.TestApp -- --models-dir models --wake-word jarvis

# Listen for "Hey Mycroft"
dotnet run --project apps/AwakeWord.TestApp -- --models-dir models --wake-word mycroft
```

### Command Line (FileProcessor)

```bash
# Process file for "Hey Jarvis"
dotnet run --project apps/AwakeWord.FileProcessor -- \
    --audio-file audio.wav \
    --models-dir models \
    --wake-word jarvis

# Process file for "Hey Mycroft"
dotnet run --project apps/AwakeWord.FileProcessor -- \
    --audio-file audio.wav \
    --models-dir models \
    --wake-word mycroft
```

### Programmatic Usage

```csharp
using AwakeWord.Core;

// Configure for "Hey Jarvis"
var jarvisConfig = new WakeWordConfig
{
    MelSpectrogramModelPath = "models/melspectrogram.onnx",
    EmbeddingModelPath = "models/embedding_model.onnx",
    WakeWordModelPath = "models/hey_jarvis_v0.1.onnx",
    WakeWord = "jarvis",
    DetectionThreshold = 0.5f
};

using var jarvisDetector = new OnnxWakeWordDetector(jarvisConfig);

// Configure for "Hey Mycroft"
var mycroftConfig = new WakeWordConfig
{
    MelSpectrogramModelPath = "models/melspectrogram.onnx",
    EmbeddingModelPath = "models/embedding_model.onnx",
    WakeWordModelPath = "models/hey_mycroft_v0.1.onnx",
    WakeWord = "mycroft",
    DetectionThreshold = 0.5f
};

using var mycroftDetector = new OnnxWakeWordDetector(mycroftConfig);

// Process audio with either detector
short[] audioChunk = /* ... */;
if (jarvisDetector.ProcessAudio(audioChunk, out var confidence))
{
    Console.WriteLine($"Hey Jarvis detected! Confidence: {confidence:0.3f}");
}
```

## Model Architecture

All wake word models use the same pipeline architecture from openWakeWord v0.5.1:

1. **Mel-Spectrogram Model** (`melspectrogram.onnx`)
   - Shared across all wake words
   - Converts raw PCM audio to mel-spectrogram features
   - Input: [batch, samples] (16-bit PCM @ 16kHz)
   - Output: [time, 1, ?, 32]

2. **Embedding Model** (`embedding_model.onnx`)
   - Shared across all wake words
   - Extracts audio embeddings from mel-spectrogram
   - Input: [batch, 76, 32, 1]
   - Output: [batch, 1, 1, 96]

3. **Wake Word Classifier** (e.g., `hey_jarvis_v0.1.onnx`, `hey_mycroft_v0.1.onnx`)
   - **Unique per wake word**
   - Classifies if the wake word is present
   - Input: [1, 16, 96]
   - Output: [1, 1] (confidence score)

## Adding New Wake Words

To add support for additional wake words:

### 1. Download the ONNX Model

Download the wake word model from [openWakeWord releases](https://github.com/dscripka/openWakeWord/releases):

```bash
# Example: Download "Hey Alexa" model
Invoke-WebRequest -Uri "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_alexa_v0.1.onnx" `
    -OutFile "models/hey_alexa_v0.1.onnx"
```

### 2. Update Application Code (Optional)

If you want command-line support, update the wake word mapping in `Program.cs`:

```csharp
// In AwakeWord.TestApp/Program.cs or AwakeWord.FileProcessor/Program.cs
var wakeWordModelPath = wakeWord.ToLowerInvariant() switch
{
    "jarvis" => Path.Combine(modelsDir, "hey_jarvis_v0.1.onnx"),
    "mycroft" => Path.Combine(modelsDir, "hey_mycroft_v0.1.onnx"),
    "alexa" => Path.Combine(modelsDir, "hey_alexa_v0.1.onnx"), // Add this line
    _ => Path.Combine(modelsDir, $"hey_{wakeWord}_v0.1.onnx")
};
```

### 3. Use the New Wake Word

```bash
# Command line
dotnet run --project apps/AwakeWord.TestApp -- --wake-word alexa

# Or programmatically
var config = new WakeWordConfig
{
    MelSpectrogramModelPath = "models/melspectrogram.onnx",
    EmbeddingModelPath = "models/embedding_model.onnx",
    WakeWordModelPath = "models/hey_alexa_v0.1.onnx",
    WakeWord = "alexa",
    DetectionThreshold = 0.5f
};
```

## Available openWakeWord Models

The openWakeWord project provides many pre-trained models. See the [releases page](https://github.com/dscripka/openWakeWord/releases) for available wake words.

Common wake words include:
- hey_jarvis
- hey_mycroft
- alexa
- hey_rhasspy
- ok_nabu
- And many more!

## Model Performance

- **Inference Time**: ~5-10ms per 80ms chunk on modern CPUs
- **Memory Usage**: ~5-10 MB per model loaded
- **Accuracy**: Varies by model and audio quality; adjust threshold as needed

## Threshold Tuning

Different wake words may require different detection thresholds:

| Wake Word | Recommended Threshold | Notes |
|-----------|----------------------|-------|
| Jarvis | 0.5 | Default setting |
| Mycroft | 0.5 | Default setting |

**Tips for tuning:**
- Start with `0.5` as baseline
- Lower to `0.3-0.4` for more detections (may increase false positives)
- Raise to `0.6-0.7` for fewer false positives (may miss some activations)
- Test with real audio in your target environment

## Troubleshooting

### Model file not found

Ensure the model file exists:
```bash
ls models/hey_mycroft_v0.1.onnx
```

### Wrong wake word detected

Make sure you're using the correct model file:
- `hey_jarvis_v0.1.onnx` only detects "Hey Jarvis"
- `hey_mycroft_v0.1.onnx` only detects "Hey Mycroft"
- Each model is trained for a specific wake word phrase

### No detections

- Check audio quality (16kHz mono, clear speech)
- Lower detection threshold: `--threshold 0.3`
- Verify the correct phrase (e.g., "Hey Jarvis" not just "Jarvis")
- Test with known good audio samples

## License

All openWakeWord models are licensed under Apache 2.0. See the [openWakeWord repository](https://github.com/dscripka/openWakeWord) for details.
