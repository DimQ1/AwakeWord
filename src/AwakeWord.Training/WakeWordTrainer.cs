namespace AwakeWord.Training;

/// <summary>
/// Progress information reported during training.
/// </summary>
public sealed class TrainingProgress
{
    public int Epoch { get; init; }
    public int TotalEpochs { get; init; }
    public float Loss { get; init; }
    public float Accuracy { get; init; }
    public float? ValLoss { get; init; }
    public float? ValAccuracy { get; init; }
    public float LearningRate { get; init; }
    public int PositiveSamples { get; init; }
    public int NegativeSamples { get; init; }
}

/// <summary>
/// Orchestrates the full wake-word training pipeline:
///   1. Extract embeddings from positive and negative audio clips
///   2. Train a binary classifier on the embedding features
///   3. Export the trained model to ONNX format
/// </summary>
public sealed class WakeWordTrainer : IDisposable
{
    private readonly TrainingConfig _config;
    private readonly EmbeddingExtractor _extractor;

    public event Action<TrainingProgress>? OnProgress;
    public event Action<string>? OnLog;

    public WakeWordTrainer(TrainingConfig config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _config.Validate();
        _extractor = new EmbeddingExtractor(
            config.MelSpectrogramModelPath,
            config.EmbeddingModelPath,
            config.FeatureFrames);
    }

    /// <summary>
    /// Train a wake-word classifier from positive and negative audio files.
    /// </summary>
    /// <param name="positiveFiles">Audio files containing the wake word.</param>
    /// <param name="negativeFiles">Audio files without the wake word (background/other speech).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Path to the exported ONNX model.</returns>
    public async Task<string> TrainAsync(
        IReadOnlyList<string> positiveFiles,
        IReadOnlyList<string> negativeFiles,
        CancellationToken cancellationToken = default)
    {
        if (positiveFiles.Count == 0)
            throw new ArgumentException("At least one positive audio file is required.", nameof(positiveFiles));
        if (negativeFiles.Count == 0)
            throw new ArgumentException("At least one negative audio file is required.", nameof(negativeFiles));

        // 1. Extract features
        Log($"Extracting features from {positiveFiles.Count} positive and {negativeFiles.Count} negative files...");

        var positiveSamples = new List<float[]>();
        var negativeSamples = new List<float[]>();

        await Task.Run(() =>
        {
            foreach (var file in positiveFiles)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Log($"  Processing positive: {Path.GetFileName(file)}");
                var windows = _extractor.ExtractFeatureWindows(file);
                positiveSamples.AddRange(windows);
                Log($"    → {windows.Count} feature windows extracted");
            }

            foreach (var file in negativeFiles)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Log($"  Processing negative: {Path.GetFileName(file)}");
                var windows = _extractor.ExtractFeatureWindows(file);
                negativeSamples.AddRange(windows);
                Log($"    → {windows.Count} feature windows extracted");
            }
        }, cancellationToken);

        if (positiveSamples.Count == 0)
            throw new InvalidOperationException("No feature windows could be extracted from positive files. Audio may be too short.");
        if (negativeSamples.Count == 0)
            throw new InvalidOperationException("No feature windows could be extracted from negative files. Audio may be too short.");

        Log($"Total samples: {positiveSamples.Count} positive, {negativeSamples.Count} negative");

        // 2. Train classifier
        Log($"Training classifier for {_config.Epochs} epochs...");
        var weights = await TrainOnFeaturesAsync(positiveSamples, negativeSamples, cancellationToken);

        // 3. Export to ONNX
        Log($"Exporting model to {_config.OutputModelPath}...");
        OnnxModelExporter.Export(weights, _config.FeatureFrames, _config.EmbeddingDim, _config.OutputModelPath);
        Log("Training complete!");

        return _config.OutputModelPath;
    }

    /// <summary>
    /// Train from raw PCM samples instead of files (useful for recorded audio).
    /// </summary>
    public async Task<string> TrainFromSamplesAsync(
        IReadOnlyList<short[]> positiveSamples,
        IReadOnlyList<short[]> negativeSamples,
        CancellationToken cancellationToken = default)
    {
        if (positiveSamples.Count == 0)
            throw new ArgumentException("At least one positive sample is required.", nameof(positiveSamples));
        if (negativeSamples.Count == 0)
            throw new ArgumentException("At least one negative sample is required.", nameof(negativeSamples));

        var positiveFeatures = new List<float[]>();
        var negativeFeatures = new List<float[]>();

        await Task.Run(() =>
        {
            foreach (var pcm in positiveSamples)
            {
                cancellationToken.ThrowIfCancellationRequested();
                positiveFeatures.AddRange(_extractor.ExtractFeatureWindows(pcm));
            }
            foreach (var pcm in negativeSamples)
            {
                cancellationToken.ThrowIfCancellationRequested();
                negativeFeatures.AddRange(_extractor.ExtractFeatureWindows(pcm));
            }
        }, cancellationToken);

        if (positiveFeatures.Count == 0)
            throw new InvalidOperationException("No features from positive samples. Audio may be too short (need >1.5s).");
        if (negativeFeatures.Count == 0)
            throw new InvalidOperationException("No features from negative samples. Audio may be too short (need >1.5s).");

        Log($"Extracted {positiveFeatures.Count} positive, {negativeFeatures.Count} negative feature windows");

        var weights = await TrainOnFeaturesAsync(positiveFeatures, negativeFeatures, cancellationToken);
        OnnxModelExporter.Export(weights, _config.FeatureFrames, _config.EmbeddingDim, _config.OutputModelPath);
        Log("Training complete!");

        return _config.OutputModelPath;
    }

    private void Log(string message) => OnLog?.Invoke(message);

    public void Dispose() => _extractor.Dispose();

    private async Task<ModelWeights> TrainOnFeaturesAsync(
        List<float[]> positiveSamples,
        List<float[]> negativeSamples,
        CancellationToken cancellationToken)
    {
        var classifier = new WakeWordClassifier(_config.FeatureFrames, _config.EmbeddingDim, _config.HiddenSize);

        var posWeight = 1f;
        var negWeight = 1f;
        if (_config.UseBalancedClassWeights)
        {
            var total = positiveSamples.Count + negativeSamples.Count;
            posWeight = total / (2f * positiveSamples.Count);
            negWeight = total / (2f * negativeSamples.Count);
        }

        var rng = new Random(42);
        var (trainSamples, valSamples) = SplitTrainValidation(positiveSamples, negativeSamples, rng, _config.ValidationSplit);

        var lr = _config.LearningRate;
        var bestMetric = float.MaxValue;
        var epochsSinceImprovement = 0;
        var useValidation = valSamples.Count > 0;

        await Task.Run(() =>
        {
            for (var epoch = 0; epoch < _config.Epochs; epoch++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                ShuffleInPlace(trainSamples, rng);

                var epochLoss = 0f;
                var correct = 0;

                foreach (var (features, label) in trainSamples)
                {
                    var trainInput = _config.InputNoiseStdDev > 0f
                        ? AddGaussianNoise(features, _config.InputNoiseStdDev, rng)
                        : features;

                    var loss = classifier.TrainStep(trainInput, label, lr, posWeight, negWeight, _config.WeightDecay);
                    epochLoss += loss;

                    var prediction = classifier.Forward(features);
                    if ((prediction >= 0.5f && label >= 0.5f) || (prediction < 0.5f && label < 0.5f))
                        correct++;
                }

                var trainLoss = epochLoss / trainSamples.Count;
                var trainAccuracy = (float)correct / trainSamples.Count;

                float? valLoss = null;
                float? valAccuracy = null;

                if (useValidation)
                {
                    Evaluate(classifier, valSamples, out var vLoss, out var vAcc);
                    valLoss = vLoss;
                    valAccuracy = vAcc;
                }

                var metric = useValidation ? valLoss!.Value : trainLoss;

                if (metric < bestMetric - _config.EarlyStoppingMinDelta)
                {
                    bestMetric = metric;
                    epochsSinceImprovement = 0;
                }
                else
                {
                    epochsSinceImprovement++;

                    if (useValidation && _config.LrDecayPatience > 0 &&
                        epochsSinceImprovement % _config.LrDecayPatience == 0)
                    {
                        lr = Math.Max(_config.MinLearningRate, lr * _config.LrDecayFactor);
                    }

                    if (useValidation && _config.EarlyStoppingPatience > 0 &&
                        epochsSinceImprovement >= _config.EarlyStoppingPatience)
                    {
                        Log($"  Early stopping at epoch {epoch + 1} (best val loss {bestMetric:F4})");
                        break;
                    }
                }

                if (epoch % 5 == 0 || epoch == _config.Epochs - 1)
                {
                    OnProgress?.Invoke(new TrainingProgress
                    {
                        Epoch = epoch + 1,
                        TotalEpochs = _config.Epochs,
                        Loss = trainLoss,
                        Accuracy = trainAccuracy,
                        ValLoss = valLoss,
                        ValAccuracy = valAccuracy,
                        LearningRate = lr,
                        PositiveSamples = positiveSamples.Count,
                        NegativeSamples = negativeSamples.Count
                    });

                    if (useValidation)
                    {
                        Log($"  Epoch {epoch + 1}/{_config.Epochs}: loss={trainLoss:F4}, acc={trainAccuracy:P1}, " +
                            $"val_loss={valLoss:F4}, val_acc={valAccuracy:P1}, lr={lr:E2}");
                    }
                    else
                    {
                        Log($"  Epoch {epoch + 1}/{_config.Epochs}: loss={trainLoss:F4}, acc={trainAccuracy:P1}, lr={lr:E2}");
                    }
                }
            }
        }, cancellationToken);

        return classifier.GetWeights();
    }

    private static void Evaluate(WakeWordClassifier classifier, List<(float[] features, float label)> samples,
        out float loss, out float accuracy)
    {
        var totalLoss = 0f;
        var correct = 0;
        const float eps = 1e-7f;

        foreach (var (features, label) in samples)
        {
            var prediction = classifier.Forward(features);
            totalLoss += -(label * MathF.Log(prediction + eps) + (1 - label) * MathF.Log(1 - prediction + eps));

            if ((prediction >= 0.5f && label >= 0.5f) || (prediction < 0.5f && label < 0.5f))
                correct++;
        }

        loss = totalLoss / samples.Count;
        accuracy = (float)correct / samples.Count;
    }

    private static (List<(float[] features, float label)> train, List<(float[] features, float label)> val)
        SplitTrainValidation(List<float[]> positiveSamples, List<float[]> negativeSamples, Random rng, float valSplit)
    {
        var pos = new List<float[]>(positiveSamples);
        var neg = new List<float[]>(negativeSamples);

        ShuffleInPlace(pos, rng);
        ShuffleInPlace(neg, rng);

        var posValCount = GetValidationCount(pos.Count, valSplit);
        var negValCount = GetValidationCount(neg.Count, valSplit);

        var val = new List<(float[] features, float label)>(posValCount + negValCount);
        var train = new List<(float[] features, float label)>(pos.Count + neg.Count - posValCount - negValCount);

        for (var i = 0; i < pos.Count; i++)
            (i < posValCount ? val : train).Add((pos[i], 1f));
        for (var i = 0; i < neg.Count; i++)
            (i < negValCount ? val : train).Add((neg[i], 0f));

        return (train, val);
    }

    private static int GetValidationCount(int count, float valSplit)
    {
        if (valSplit <= 0f || count < 2)
            return 0;

        var valCount = (int)MathF.Round(count * valSplit);
        if (valCount <= 0)
            valCount = 1;
        if (valCount >= count)
            valCount = count / 2;

        return valCount;
    }

    private static void ShuffleInPlace<T>(List<T> list, Random rng)
    {
        for (var i = list.Count - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private static float[] AddGaussianNoise(float[] input, float stdDev, Random rng)
    {
        var noisy = new float[input.Length];
        for (var i = 0; i < input.Length; i++)
            noisy[i] = input[i] + (float)NextGaussian(rng) * stdDev;
        return noisy;
    }

    private static double NextGaussian(Random rng)
    {
        var u1 = 1.0 - rng.NextDouble();
        var u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
