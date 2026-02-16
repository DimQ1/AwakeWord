namespace AwakeWord.Core;

/// <summary>
/// Configuration for the openWakeWord pipeline detector.
/// </summary>
public sealed class WakeWordConfig
{
    /// <summary>The wake word to detect (default: "jarvis").</summary>
    public string WakeWord { get; set; } = "jarvis";

    /// <summary>Detection threshold (0–1). Higher = fewer false positives.</summary>
    public float DetectionThreshold { get; set; } = 0.5f;

    /// <summary>Window size for averaging confidence scores (default: 5).</summary>
    public int SmoothingWindow { get; set; } = 5;

    /// <summary>Number of consecutive averaged hits required to trigger (default: 2).</summary>
    public int MinConsecutiveDetections { get; set; } = 2;

    /// <summary>Path to the mel-spectrogram ONNX model (melspectrogram.onnx).</summary>
    public string MelSpectrogramModelPath { get; set; } = string.Empty;

    /// <summary>Path to the embedding ONNX model (embedding_model.onnx).</summary>
    public string EmbeddingModelPath { get; set; } = string.Empty;

    /// <summary>Path to the wake-word classifier ONNX model (e.g., hey_jarvis_v0.1.onnx).</summary>
    public string WakeWordModelPath { get; set; } = string.Empty;

    /// <summary>Number of embedding frames the wake-word model expects (default: 16).</summary>
    public int FeatureFrames { get; set; } = 16;

    /// <summary>Number of inter-op threads for ONNX Runtime (default: 1).</summary>
    public int NumThreads { get; set; } = 1;

    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(WakeWord))
            throw new ArgumentException("Wake word must be provided.", nameof(WakeWord));

        if (DetectionThreshold is < 0f or > 1f)
            throw new ArgumentOutOfRangeException(nameof(DetectionThreshold), "Threshold must be between 0 and 1.");

        if (string.IsNullOrWhiteSpace(MelSpectrogramModelPath))
            throw new ArgumentException("Mel-spectrogram model path must be provided.", nameof(MelSpectrogramModelPath));

        if (string.IsNullOrWhiteSpace(EmbeddingModelPath))
            throw new ArgumentException("Embedding model path must be provided.", nameof(EmbeddingModelPath));

        if (string.IsNullOrWhiteSpace(WakeWordModelPath))
            throw new ArgumentException("Wake-word model path must be provided.", nameof(WakeWordModelPath));

        if (FeatureFrames <= 0)
            throw new ArgumentOutOfRangeException(nameof(FeatureFrames), "Feature frames must be positive.");

        if (SmoothingWindow <= 0)
            throw new ArgumentOutOfRangeException(nameof(SmoothingWindow), "Smoothing window must be positive.");

        if (MinConsecutiveDetections <= 0)
            throw new ArgumentOutOfRangeException(nameof(MinConsecutiveDetections), "Min consecutive detections must be positive.");
    }
}
