using AwakeWord.Core;

static string? GetArg(string[] args, string name)
{
    for (var i = 0; i < args.Length - 1; i++)
    {
        if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
            return args[i + 1];
    }
    return null;
}

static float GetFloatArg(string[] args, string name, float fallback)
{
    var value = GetArg(args, name);
    return float.TryParse(value, out var parsed) ? parsed : fallback;
}

var modelsDir = GetArg(args, "--models-dir") ?? "models";
var audioFile = GetArg(args, "--audio-file");
var wakeWord = GetArg(args, "--wake-word") ?? "jarvis";
var threshold = GetFloatArg(args, "--threshold", 0.5f);

if (string.IsNullOrEmpty(audioFile))
{
    Console.WriteLine("AwakeWord File Processor");
    Console.WriteLine("Processes audio files (WAV, MP3, M4A, etc.) for wake-word detection.");
    Console.WriteLine();
    Console.WriteLine("Usage:");
    Console.WriteLine("  AwakeWord.FileProcessor --audio-file <path> [--models-dir <path>] [--wake-word <word>] [--threshold <0-1>]");
    Console.WriteLine();
    Console.WriteLine("Arguments:");
    Console.WriteLine("  --audio-file    Path to audio file (WAV, MP3, M4A, etc.)");
    Console.WriteLine("  --models-dir    Path to models directory (default: 'models')");
    Console.WriteLine("  --wake-word     Wake word to detect: jarvis, mycroft (default: jarvis)");
    Console.WriteLine("  --threshold     Detection threshold 0-1 (default: 0.5)");
    Console.WriteLine();
    Console.WriteLine("Supported formats: WAV, MP3, M4A, AIFF, WMA");
    return;
}

if (!File.Exists(audioFile))
{
    Console.WriteLine($"Error: Audio file not found: {audioFile}");
    return;
}

var extension = Path.GetExtension(audioFile);
if (!AudioFileLoader.IsSupportedFormat(extension))
{
    Console.WriteLine($"Warning: File format '{extension}' may not be supported.");
    Console.WriteLine("Supported formats: WAV, MP3, M4A, AIFF, WMA");
}

var melspecPath = Path.Combine(modelsDir, "melspectrogram.onnx");
var embeddingPath = Path.Combine(modelsDir, "embedding_model.onnx");

// Map wake word to model file
var wakeWordModelPath = wakeWord.ToLowerInvariant() switch
{
    "jarvis" => Path.Combine(modelsDir, "hey_jarvis_v0.1.onnx"),
    "mycroft" => Path.Combine(modelsDir, "hey_mycroft_v0.1.onnx"),
    _ => Path.Combine(modelsDir, $"hey_{wakeWord}_v0.1.onnx")
};

if (!File.Exists(melspecPath) || !File.Exists(embeddingPath) || !File.Exists(wakeWordModelPath))
{
    Console.WriteLine("Error: One or more ONNX model files not found.");
    Console.WriteLine($"Expected files in '{modelsDir}':");
    Console.WriteLine($"  - melspectrogram.onnx");
    Console.WriteLine($"  - embedding_model.onnx");
    Console.WriteLine($"  - {Path.GetFileName(wakeWordModelPath)}");
    Console.WriteLine();
    Console.WriteLine("Supported wake words: jarvis, mycroft");
    return;
}

var config = new WakeWordConfig
{
    MelSpectrogramModelPath = melspecPath,
    EmbeddingModelPath = embeddingPath,
    WakeWordModelPath = wakeWordModelPath,
    WakeWord = wakeWord,
    DetectionThreshold = threshold
};

Console.WriteLine($"Loading models from '{modelsDir}'...");
using var detector = new OnnxWakeWordDetector(config);
Console.WriteLine($"Models loaded successfully.");
Console.WriteLine();

Console.WriteLine($"Loading audio file: {audioFile}");
var samples = AudioFileLoader.LoadAudioFile(audioFile);
var durationSeconds = samples.Length / (float)AudioFormat.SampleRateHz;
Console.WriteLine($"Loaded {samples.Length:N0} samples ({durationSeconds:F2} seconds)");
Console.WriteLine($"Detection threshold: {threshold:0.00}");
Console.WriteLine();

Console.WriteLine("Processing audio...");
const int chunkSize = 1280; // 80ms chunks
var detections = new List<(float timeMs, float confidence)>();
var maxConfidenceOverall = float.MinValue;

for (var offset = 0; offset < samples.Length; offset += chunkSize)
{
    var length = Math.Min(chunkSize, samples.Length - offset);
    var chunk = new short[length];
    Array.Copy(samples, offset, chunk, 0, length);

    detector.ProcessAudio(chunk, out var confidence);
    maxConfidenceOverall = Math.Max(maxConfidenceOverall, confidence);

    if (confidence >= threshold)
    {
        var timeMs = offset * 1000.0f / AudioFormat.SampleRateHz;
        detections.Add((timeMs, confidence));
    }
}

Console.WriteLine();
Console.WriteLine($"Max confidence across all chunks: {maxConfidenceOverall:E4}");

if (detections.Count > 0)
{
    Console.WriteLine($"✓ Wake word detected {detections.Count} time(s):");
    foreach (var (timeMs, confidence) in detections)
    {
        Console.WriteLine($"  {timeMs / 1000.0:F2}s - confidence {confidence:F3}");
    }
}
else
{
    Console.WriteLine("No wake word detections found.");
    Console.WriteLine("Try lowering the threshold with --threshold 0.3 or use an audio file containing the wake word.");
}
