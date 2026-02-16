using AwakeWord.Core;

namespace AwakeWord.Core.Tests;

[TestClass]
public sealed class WakeWordConfigTests
{
    [TestMethod]
    public void DefaultWakeWordIsJarvis()
    {
        var config = new WakeWordConfig();
        Assert.AreEqual("jarvis", config.WakeWord);
    }

    [TestMethod]
    public void ValidateRequiresMelSpectrogramModelPath()
    {
        var config = new WakeWordConfig
        {
            MelSpectrogramModelPath = "",
            EmbeddingModelPath = "embedding.onnx",
            WakeWordModelPath = "wakeword.onnx"
        };
        Assert.ThrowsExactly<ArgumentException>(() => config.Validate());
    }

    [TestMethod]
    public void ValidateRequiresEmbeddingModelPath()
    {
        var config = new WakeWordConfig
        {
            MelSpectrogramModelPath = "mel.onnx",
            EmbeddingModelPath = "",
            WakeWordModelPath = "wakeword.onnx"
        };
        Assert.ThrowsExactly<ArgumentException>(() => config.Validate());
    }

    [TestMethod]
    public void ValidateRequiresWakeWordModelPath()
    {
        var config = new WakeWordConfig
        {
            MelSpectrogramModelPath = "mel.onnx",
            EmbeddingModelPath = "embedding.onnx",
            WakeWordModelPath = ""
        };
        Assert.ThrowsExactly<ArgumentException>(() => config.Validate());
    }

    [TestMethod]
    public void ValidateRejectsInvalidThreshold()
    {
        var config = new WakeWordConfig
        {
            MelSpectrogramModelPath = "mel.onnx",
            EmbeddingModelPath = "embedding.onnx",
            WakeWordModelPath = "wakeword.onnx",
            DetectionThreshold = 1.5f
        };
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => config.Validate());
    }

    [TestMethod]
    public void ValidateAcceptsCompleteConfig()
    {
        var config = new WakeWordConfig
        {
            MelSpectrogramModelPath = "mel.onnx",
            EmbeddingModelPath = "embedding.onnx",
            WakeWordModelPath = "wakeword.onnx"
        };
        config.Validate(); // should not throw
    }

    [TestMethod]
    public void DefaultFeatureFramesIs16()
    {
        var config = new WakeWordConfig();
        Assert.AreEqual(16, config.FeatureFrames);
    }

    [TestMethod]
    public void DefaultThresholdIs0Point5()
    {
        var config = new WakeWordConfig();
        Assert.AreEqual(0.5f, config.DetectionThreshold);
    }
}

[TestClass]
public sealed class AudioFormatTests
{
    [TestMethod]
    public void EnsureAcceptsExpectedFormat()
    {
        AudioFormat.Ensure(AudioFormat.SampleRateHz, AudioFormat.Channels);
    }

    [TestMethod]
    public void EnsureRejectsUnexpectedFormat()
    {
        Assert.ThrowsExactly<ArgumentException>(() => AudioFormat.Ensure(8000, AudioFormat.Channels));
        Assert.ThrowsExactly<ArgumentException>(() => AudioFormat.Ensure(AudioFormat.SampleRateHz, 2));
    }
}

[TestClass]
public sealed class OnnxWakeWordDetectorTests
{
    private static string ModelsDir => Path.Combine(
        Path.GetDirectoryName(typeof(OnnxWakeWordDetectorTests).Assembly.Location)!,
        "..", "..", "..", "..", "..", "models");

    private static bool ModelsExist =>
        File.Exists(Path.Combine(ModelsDir, "melspectrogram.onnx")) &&
        File.Exists(Path.Combine(ModelsDir, "embedding_model.onnx")) &&
        File.Exists(Path.Combine(ModelsDir, "hey_jarvis_v0.1.onnx"));

    private WakeWordConfig CreateConfig() => new()
    {
        MelSpectrogramModelPath = Path.Combine(ModelsDir, "melspectrogram.onnx"),
        EmbeddingModelPath = Path.Combine(ModelsDir, "embedding_model.onnx"),
        WakeWordModelPath = Path.Combine(ModelsDir, "hey_jarvis_v0.1.onnx"),
        WakeWord = "jarvis",
        DetectionThreshold = 0.5f
    };

    [TestMethod]
    public void CanCreateDetector()
    {
        if (!ModelsExist) Assert.Inconclusive("ONNX models not found; skipping.");

        using var detector = new OnnxWakeWordDetector(CreateConfig());
        Assert.AreEqual("jarvis", detector.WakeWord);
    }

    [TestMethod]
    public void SilenceDoesNotTrigger()
    {
        if (!ModelsExist) Assert.Inconclusive("ONNX models not found; skipping.");

        using var detector = new OnnxWakeWordDetector(CreateConfig());

        // Feed 2 seconds of silence
        var silence = new short[1280]; // 80 ms
        for (var i = 0; i < 25; i++) // ~2 seconds
        {
            var detected = detector.ProcessAudio(silence, out var confidence);
            Assert.IsFalse(detected, $"Silence should not trigger detection (confidence {confidence:0.000}).");
        }
    }

    [TestMethod]
    public void ResetClearsState()
    {
        if (!ModelsExist) Assert.Inconclusive("ONNX models not found; skipping.");

        using var detector = new OnnxWakeWordDetector(CreateConfig());

        // Feed some noise, then reset
        var rng = new Random(42);
        var noise = new short[1280];
        for (var i = 0; i < noise.Length; i++)
            noise[i] = (short)rng.Next(-5000, 5000);

        detector.ProcessAudio(noise, out _);
        detector.Reset();

        // After reset, silence should still not trigger
        var silence = new short[1280];
        for (var i = 0; i < 20; i++)
        {
            var detected = detector.ProcessAudio(silence, out _);
            Assert.IsFalse(detected);
        }
    }
}
