using AwakeWord.Core;
using NAudio.Wave;

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
var wakeWord = GetArg(args, "--wake-word") ?? "jarvis";
var threshold = GetFloatArg(args, "--threshold", 0.5f);

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
    Console.WriteLine("One or more ONNX model files not found in the models directory.");
    Console.WriteLine($"Expected:");
    Console.WriteLine($"  - {melspecPath}");
    Console.WriteLine($"  - {embeddingPath}");
    Console.WriteLine($"  - {wakeWordModelPath}");
    Console.WriteLine();
    Console.WriteLine("Usage: AwakeWord.TestApp [--models-dir <path>] [--wake-word <word>] [--threshold <0-1>]");
    Console.WriteLine();
    Console.WriteLine("Supported wake words:");
    Console.WriteLine("  - jarvis (default)");
    Console.WriteLine("  - mycroft");
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
Console.WriteLine($"Listening for wake word '{detector.WakeWord}' (threshold {threshold:0.00}). Press Enter to stop.");
Console.WriteLine();

using var waveIn = new WaveInEvent
{
    WaveFormat = new WaveFormat(AudioFormat.SampleRateHz, 16, AudioFormat.Channels),
    BufferMilliseconds = 80
};

var lastFeedback = DateTime.MinValue;
var maxConfidenceSinceLastFeedback = 0f;

waveIn.DataAvailable += (_, eventArgs) =>
{
    var sampleCount = eventArgs.BytesRecorded / 2;
    var samples = new short[sampleCount];
    for (var i = 0; i < sampleCount; i++)
        samples[i] = (short)(eventArgs.Buffer[i * 2] | (eventArgs.Buffer[i * 2 + 1] << 8));

    detector.ProcessAudio(samples, out var confidence);
    maxConfidenceSinceLastFeedback = Math.Max(maxConfidenceSinceLastFeedback, confidence);

    if (confidence >= threshold)
    {
        Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] *** Wake word '{detector.WakeWord}' detected! *** (confidence {confidence:0.000})");
        maxConfidenceSinceLastFeedback = 0f;
        lastFeedback = DateTime.Now;
    }
    else if ((DateTime.Now - lastFeedback).TotalSeconds >= 2)
    {
        Console.Write($"\r  Listening... max confidence: {maxConfidenceSinceLastFeedback:0.000}   ");
        maxConfidenceSinceLastFeedback = 0f;
        lastFeedback = DateTime.Now;
    }
};

waveIn.StartRecording();
Console.ReadLine();
waveIn.StopRecording();
