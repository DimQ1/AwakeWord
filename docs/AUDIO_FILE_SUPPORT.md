# Audio File Support

The AwakeWord library supports loading and processing audio files in multiple formats including **WAV**, **MP3**, **M4A**, **AIFF**, and **WMA**.

## AudioFileLoader API

The `AudioFileLoader` class provides a simple API for loading audio files:

```csharp
using AwakeWord.Core;

// Load an audio file (WAV, MP3, etc.)
short[] samples = AudioFileLoader.LoadAudioFile("path/to/audio.mp3");

// Check if a format is supported
bool isSupported = AudioFileLoader.IsSupportedFormat(".mp3"); // true
```

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV    | .wav      | Native support, optimized for 16-bit PCM |
| MP3    | .mp3      | Full support via NAudio |
| M4A    | .m4a      | AAC audio container |
| AIFF   | .aiff, .aif | Apple audio format |
| WMA    | .wma      | Windows Media Audio |

## Automatic Format Conversion

All audio files are automatically converted to the required format for wake-word detection:
- **Sample Rate**: 16 kHz
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit PCM
- **Byte Order**: Little-endian

The conversion happens transparently using NAudio's `AudioFileReader`, which leverages Windows Media Foundation for decoding.

## Example: Processing MP3 Files

```csharp
using AwakeWord.Core;

var config = new WakeWordConfig
{
    MelSpectrogramModelPath = "models/melspectrogram.onnx",
    EmbeddingModelPath = "models/embedding_model.onnx",
    WakeWordModelPath = "models/hey_jarvis_v0.1.onnx",
    WakeWord = "jarvis",
    DetectionThreshold = 0.5f
};

using var detector = new OnnxWakeWordDetector(config);

// Load MP3 file
var samples = AudioFileLoader.LoadAudioFile("audio.mp3");

// Process in chunks
const int chunkSize = 1280; // 80ms at 16kHz
for (var offset = 0; offset < samples.Length; offset += chunkSize)
{
    var length = Math.Min(chunkSize, samples.Length - offset);
    var chunk = new short[length];
    Array.Copy(samples, offset, chunk, 0, length);

    if (detector.ProcessAudio(chunk, out var confidence))
    {
        var timeSeconds = offset / 16000.0;
        Console.WriteLine($"Wake word detected at {timeSeconds:0.2f}s (confidence: {confidence:0.3f})");
    }
}
```

## File Processor Application

The library includes a command-line tool for processing audio files:

```bash
# Process a WAV file
dotnet run --project apps/AwakeWord.FileProcessor -- \
    --audio-file input.wav \
    --models-dir models \
    --threshold 0.5

# Process an MP3 file
dotnet run --project apps/AwakeWord.FileProcessor -- \
    --audio-file input.mp3 \
    --models-dir models \
    --threshold 0.5
```

### FileProcessor Options

- `--audio-file` - Path to audio file (required)
- `--models-dir` - Directory containing ONNX models (default: `models`)
- `--threshold` - Detection threshold 0-1 (default: `0.5`)

## Error Handling

The `AudioFileLoader` provides clear exceptions:

```csharp
try
{
    var samples = AudioFileLoader.LoadAudioFile("audio.mp3");
}
catch (FileNotFoundException ex)
{
    Console.WriteLine($"File not found: {ex.Message}");
}
catch (InvalidOperationException ex)
{
    Console.WriteLine($"Failed to load audio: {ex.Message}");
}
```

## Performance Notes

- **WAV files** in the exact target format (16-bit PCM mono @ 16kHz) are loaded directly without conversion for optimal performance
- **Other formats** and non-matching WAV files use NAudio's conversion pipeline, which is still quite fast
- Large files are processed in streaming fashion to minimize memory usage

## Platform Requirements

Audio file support requires Windows Media Foundation (WMF), which is available on:
- Windows 7 and later
- Windows Server 2008 R2 and later

For cross-platform support on Linux/macOS, you may need to implement a custom audio loader using a different decoding library (e.g., FFmpeg bindings).
