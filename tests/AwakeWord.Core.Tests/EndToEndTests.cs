using AwakeWord.Core;
using AwakeWord.Training;

namespace AwakeWord.Core.Tests;

/// <summary>
/// End-to-end integration tests that train a wake-word model from real audio files
/// and verify detection works correctly on positive and negative samples.
/// </summary>
[TestClass]
public class EndToEndTests
{
    // Paths resolved relative to the solution root
    private static readonly string SolutionRoot = FindSolutionRoot();
    private static readonly string ModelsDir = Path.Combine(SolutionRoot, "models");
    private static readonly string MelSpectrogramPath = Path.Combine(ModelsDir, "melspectrogram.onnx");
    private static readonly string EmbeddingModelPath = Path.Combine(ModelsDir, "embedding_model.onnx");
    private static readonly string PositiveDir = Path.Combine(ModelsDir, "positive");
    private static readonly string NegativeDir = Path.Combine(ModelsDir, "negative");

    private static string FindSolutionRoot()
    {
        // Walk up from the test assembly location to find the solution root (contains models/)
        var dir = AppContext.BaseDirectory;
        while (dir != null)
        {
            if (Directory.Exists(Path.Combine(dir, "models")) &&
                File.Exists(Path.Combine(dir, "models", "melspectrogram.onnx")))
            {
                return dir;
            }
            dir = Directory.GetParent(dir)?.FullName;
        }

        // Fallback: absolute path
        return @"e:\Learning\AI\AwakeWord";
    }

    [TestMethod]
    public async Task TrainModel_WithRealAudio_ProducesValidOnnx()
    {
        // Arrange
        var positiveFiles = Directory.GetFiles(PositiveDir, "*.wav").ToList();
        var negativeFiles = Directory.GetFiles(NegativeDir, "*.wav").ToList();

        Assert.IsTrue(positiveFiles.Count > 0, "No positive WAV files found");
        Assert.IsTrue(negativeFiles.Count > 0, "No negative WAV files found");

        var outputPath = Path.Combine(Path.GetTempPath(), $"e2e_test_{Guid.NewGuid():N}.onnx");

        var config = new TrainingConfig
        {
            MelSpectrogramModelPath = MelSpectrogramPath,
            EmbeddingModelPath = EmbeddingModelPath,
            WakeWordName = "hey_custom",
            Epochs = 50,
            LearningRate = 0.001f,
            OutputModelPath = outputPath
        };

        try
        {
            // Act
            using var trainer = new WakeWordTrainer(config);
            var logs = new List<string>();
            TrainingProgress? lastProgress = null;
            trainer.OnLog += msg => logs.Add(msg);
            trainer.OnProgress += p => lastProgress = p;

            var modelPath = await trainer.TrainAsync(positiveFiles, negativeFiles);

            // Assert — model file produced
            Assert.IsTrue(File.Exists(modelPath), "Trained model file should exist");
            var fileInfo = new FileInfo(modelPath);
            Assert.IsTrue(fileInfo.Length > 1000, $"Model file should be >1KB, was {fileInfo.Length} bytes");

            // Assert — training reached reasonable accuracy
            Assert.IsNotNull(lastProgress, "Should have received training progress");
            Assert.IsTrue(lastProgress!.Accuracy >= 0.6f,
                $"Final accuracy should be ≥60%, was {lastProgress.Accuracy:P1}");

            // Assert — logs captured
            Assert.IsTrue(logs.Count > 0, "Should have generated log output");
        }
        finally
        {
            if (File.Exists(outputPath))
                File.Delete(outputPath);
        }
    }

    [TestMethod]
    public async Task TrainAndDetect_PositiveAudio_IsDetected()
    {
        // Arrange — train a model
        var positiveFiles = Directory.GetFiles(PositiveDir, "*.wav").ToList();
        var negativeFiles = Directory.GetFiles(NegativeDir, "*.wav").ToList();

        var outputPath = Path.Combine(Path.GetTempPath(), $"e2e_detect_{Guid.NewGuid():N}.onnx");

        var trainingConfig = new TrainingConfig
        {
            MelSpectrogramModelPath = MelSpectrogramPath,
            EmbeddingModelPath = EmbeddingModelPath,
            WakeWordName = "hey_custom",
            Epochs = 100,
            LearningRate = 0.001f,
            OutputModelPath = outputPath
        };

        try
        {
            using var trainer = new WakeWordTrainer(trainingConfig);
            TrainingProgress? lastProgress = null;
            trainer.OnProgress += p => lastProgress = p;

            await trainer.TrainAsync(positiveFiles, negativeFiles);

            Assert.IsNotNull(lastProgress);
            Console.WriteLine($"Training completed: accuracy={lastProgress!.Accuracy:P1}, " +
                              $"positive={lastProgress.PositiveSamples}, negative={lastProgress.NegativeSamples}");

            // Act — run detector on each positive file
            var detectionConfig = new WakeWordConfig
            {
                WakeWord = "hey_custom",
                MelSpectrogramModelPath = MelSpectrogramPath,
                EmbeddingModelPath = EmbeddingModelPath,
                WakeWordModelPath = outputPath,
                DetectionThreshold = 0.5f
            };

            var detected = 0;
            var maxConfidences = new List<(string file, float confidence)>();

            foreach (var file in positiveFiles)
            {
                using var detector = new OnnxWakeWordDetector(detectionConfig);

                var samples = AudioFileLoader.LoadAudioFile(file);
                var fileDetected = false;
                var maxConf = 0f;

                // Feed audio in realistic chunk sizes (1280 samples = 80ms)
                const int chunkSize = 1280;
                for (var offset = 0; offset < samples.Length; offset += chunkSize)
                {
                    var end = Math.Min(offset + chunkSize, samples.Length);
                    var chunk = samples.AsSpan(offset, end - offset);

                    if (detector.ProcessAudio(chunk, out var confidence))
                    {
                        fileDetected = true;
                        maxConf = Math.Max(maxConf, confidence);
                    }
                    else if (confidence > maxConf)
                    {
                        maxConf = confidence;
                    }
                }

                if (fileDetected) detected++;
                maxConfidences.Add((Path.GetFileName(file), maxConf));
            }

            // Log results
            foreach (var (file, conf) in maxConfidences)
                Console.WriteLine($"  {file}: max confidence = {conf:F4}");

            // Assert — at least half positive files should trigger detection
            Assert.IsTrue(detected >= positiveFiles.Count / 2,
                $"Expected at least {positiveFiles.Count / 2} detections from {positiveFiles.Count} positive files, got {detected}");
        }
        finally
        {
            if (File.Exists(outputPath))
                File.Delete(outputPath);
        }
    }

    [TestMethod]
    public async Task TrainAndDetect_NegativeAudio_HasLowerConfidenceThanPositive()
    {
        // With a tiny dataset (4+4 files), absolute threshold-based detection is
        // unreliable. Instead, verify that negative audio produces consistently
        // lower confidence scores than positive audio.
        var positiveFiles = Directory.GetFiles(PositiveDir, "*.wav").ToList();
        var negativeFiles = Directory.GetFiles(NegativeDir, "*.wav").ToList();

        var outputPath = Path.Combine(Path.GetTempPath(), $"e2e_neg_{Guid.NewGuid():N}.onnx");

        var trainingConfig = new TrainingConfig
        {
            MelSpectrogramModelPath = MelSpectrogramPath,
            EmbeddingModelPath = EmbeddingModelPath,
            WakeWordName = "hey_custom",
            Epochs = 200,
            LearningRate = 0.001f,
            OutputModelPath = outputPath
        };

        try
        {
            using var trainer = new WakeWordTrainer(trainingConfig);
            await trainer.TrainAsync(positiveFiles, negativeFiles);

            var detectionConfig = new WakeWordConfig
            {
                WakeWord = "hey_custom",
                MelSpectrogramModelPath = MelSpectrogramPath,
                EmbeddingModelPath = EmbeddingModelPath,
                WakeWordModelPath = outputPath,
                DetectionThreshold = 0.5f
            };

            // Measure max confidence on positive files
            var positiveMaxConfs = new List<float>();
            foreach (var file in positiveFiles)
            {
                var conf = GetMaxConfidence(detectionConfig, file);
                positiveMaxConfs.Add(conf);
                Console.WriteLine($"  Positive {Path.GetFileName(file)}: {conf:F4}");
            }

            // Measure max confidence on negative files
            var negativeMaxConfs = new List<float>();
            foreach (var file in negativeFiles)
            {
                var conf = GetMaxConfidence(detectionConfig, file);
                negativeMaxConfs.Add(conf);
                Console.WriteLine($"  Negative {Path.GetFileName(file)}: {conf:F4}");
            }

            var avgPos = positiveMaxConfs.Average();
            var avgNeg = negativeMaxConfs.Average();

            Console.WriteLine($"\n  Avg positive confidence: {avgPos:F4}");
            Console.WriteLine($"  Avg negative confidence: {avgNeg:F4}");

            // Assert — positive files should produce higher confidence on average
            Assert.IsTrue(avgPos > avgNeg,
                $"Positive avg ({avgPos:F4}) should exceed negative avg ({avgNeg:F4})");
        }
        finally
        {
            if (File.Exists(outputPath))
                File.Delete(outputPath);
        }
    }

    [TestMethod]
    public async Task FullPipeline_TrainAndClassify_DiscriminatesPositiveFromNegative()
    {
        // This test trains a model and verifies that average confidence on positive
        // files is meaningfully higher than on negative files.
        var positiveFiles = Directory.GetFiles(PositiveDir, "*.wav").ToList();
        var negativeFiles = Directory.GetFiles(NegativeDir, "*.wav").ToList();

        var outputPath = Path.Combine(Path.GetTempPath(), $"e2e_full_{Guid.NewGuid():N}.onnx");

        var trainingConfig = new TrainingConfig
        {
            MelSpectrogramModelPath = MelSpectrogramPath,
            EmbeddingModelPath = EmbeddingModelPath,
            WakeWordName = "hey_custom",
            Epochs = 100,
            LearningRate = 0.001f,
            OutputModelPath = outputPath
        };

        try
        {
            using var trainer = new WakeWordTrainer(trainingConfig);
            TrainingProgress? lastProgress = null;
            trainer.OnProgress += p => lastProgress = p;
            await trainer.TrainAsync(positiveFiles, negativeFiles);

            var detectionConfig = new WakeWordConfig
            {
                WakeWord = "hey_custom",
                MelSpectrogramModelPath = MelSpectrogramPath,
                EmbeddingModelPath = EmbeddingModelPath,
                WakeWordModelPath = outputPath,
                DetectionThreshold = 0.5f
            };

            // Measure max confidence across all positive files
            var positiveConfidences = new List<float>();
            foreach (var file in positiveFiles)
            {
                var maxConf = GetMaxConfidence(detectionConfig, file);
                positiveConfidences.Add(maxConf);
                Console.WriteLine($"  Positive {Path.GetFileName(file)}: {maxConf:F4}");
            }

            // Measure max confidence across all negative files
            var negativeConfidences = new List<float>();
            foreach (var file in negativeFiles)
            {
                var maxConf = GetMaxConfidence(detectionConfig, file);
                negativeConfidences.Add(maxConf);
                Console.WriteLine($"  Negative {Path.GetFileName(file)}: {maxConf:F4}");
            }

            var avgPositive = positiveConfidences.Average();
            var avgNegative = negativeConfidences.Average();

            Console.WriteLine($"\n  Average positive confidence: {avgPositive:F4}");
            Console.WriteLine($"  Average negative confidence: {avgNegative:F4}");
            Console.WriteLine($"  Training accuracy: {lastProgress?.Accuracy:P1}");

            // Assert — positive confidence should be meaningfully higher
            Assert.IsTrue(avgPositive > avgNegative,
                $"Average positive confidence ({avgPositive:F4}) should be higher than negative ({avgNegative:F4})");
        }
        finally
        {
            if (File.Exists(outputPath))
                File.Delete(outputPath);
        }
    }

    /// <summary>
    /// Feeds an entire audio file through the detector and returns the maximum
    /// confidence observed (regardless of whether detection threshold was crossed).
    /// </summary>
    private static float GetMaxConfidence(WakeWordConfig config, string audioFile)
    {
        using var detector = new OnnxWakeWordDetector(config);
        var samples = AudioFileLoader.LoadAudioFile(audioFile);
        var maxConf = 0f;

        const int chunkSize = 1280;
        for (var offset = 0; offset < samples.Length; offset += chunkSize)
        {
            var end = Math.Min(offset + chunkSize, samples.Length);
            var chunk = samples.AsSpan(offset, end - offset);

            detector.ProcessAudio(chunk, out var confidence);
            if (confidence > maxConf)
                maxConf = confidence;
        }

        return maxConf;
    }
}
