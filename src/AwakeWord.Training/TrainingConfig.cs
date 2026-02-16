namespace AwakeWord.Training;

/// <summary>
/// Configuration for training a custom wake-word classifier model.
/// </summary>
public sealed class TrainingConfig
{
    /// <summary>Path to the melspectrogram.onnx model.</summary>
    public string MelSpectrogramModelPath { get; set; } = string.Empty;

    /// <summary>Path to the embedding_model.onnx model.</summary>
    public string EmbeddingModelPath { get; set; } = string.Empty;

    /// <summary>Name of the wake word being trained (e.g. "hey_custom").</summary>
    public string WakeWordName { get; set; } = string.Empty;

    /// <summary>Number of embedding frames the classifier expects (default: 16).</summary>
    public int FeatureFrames { get; set; } = 16;

    /// <summary>Embedding dimension from the embedding model (default: 96).</summary>
    public int EmbeddingDim { get; set; } = 96;

    /// <summary>Number of training epochs (default: 100).</summary>
    public int Epochs { get; set; } = 100;

    /// <summary>Learning rate for gradient descent (default: 0.0001).</summary>
    public float LearningRate { get; set; } = 0.0001f;

    /// <summary>Minimum learning rate when using decay (default: 0.000001).</summary>
    public float MinLearningRate { get; set; } = 0.000001f;

    /// <summary>Apply class balancing weights based on sample counts.</summary>
    public bool UseBalancedClassWeights { get; set; } = true;

    /// <summary>L2 weight decay to improve generalization (default: 1e-4).</summary>
    public float WeightDecay { get; set; } = 1e-4f;

    /// <summary>Standard deviation of Gaussian noise added to features during training (default: 0.01).</summary>
    public float InputNoiseStdDev { get; set; } = 0.01f;

    /// <summary>Fraction of samples held out for validation (default: 0.1).</summary>
    public float ValidationSplit { get; set; } = 0.1f;

    /// <summary>Minimum validation loss improvement to reset early stopping (default: 1e-4).</summary>
    public float EarlyStoppingMinDelta { get; set; } = 1e-4f;

    /// <summary>Stop training if no validation improvement for this many epochs (default: 12).</summary>
    public int EarlyStoppingPatience { get; set; } = 12;

    /// <summary>Reduce learning rate if no validation improvement for this many epochs (default: 4).</summary>
    public int LrDecayPatience { get; set; } = 4;

    /// <summary>Learning rate multiplier when decaying (default: 0.5).</summary>
    public float LrDecayFactor { get; set; } = 0.5f;

    /// <summary>Hidden layer width for the classifier (default: 192).</summary>
    public int HiddenSize { get; set; } = 192;

    /// <summary>Path where the trained ONNX model will be saved.</summary>
    public string OutputModelPath { get; set; } = string.Empty;

    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(MelSpectrogramModelPath))
            throw new ArgumentException("Mel-spectrogram model path is required.", nameof(MelSpectrogramModelPath));

        if (string.IsNullOrWhiteSpace(EmbeddingModelPath))
            throw new ArgumentException("Embedding model path is required.", nameof(EmbeddingModelPath));

        if (string.IsNullOrWhiteSpace(WakeWordName))
            throw new ArgumentException("Wake word name is required.", nameof(WakeWordName));

        if (string.IsNullOrWhiteSpace(OutputModelPath))
            throw new ArgumentException("Output model path is required.", nameof(OutputModelPath));

        if (FeatureFrames <= 0)
            throw new ArgumentOutOfRangeException(nameof(FeatureFrames), "Must be positive.");

        if (Epochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(Epochs), "Must be positive.");

        if (LearningRate <= 0f)
            throw new ArgumentOutOfRangeException(nameof(LearningRate), "Must be positive.");

        if (MinLearningRate <= 0f)
            throw new ArgumentOutOfRangeException(nameof(MinLearningRate), "Must be positive.");

        if (WeightDecay < 0f)
            throw new ArgumentOutOfRangeException(nameof(WeightDecay), "Must be non-negative.");

        if (InputNoiseStdDev < 0f)
            throw new ArgumentOutOfRangeException(nameof(InputNoiseStdDev), "Must be non-negative.");

        if (ValidationSplit is < 0f or > 0.5f)
            throw new ArgumentOutOfRangeException(nameof(ValidationSplit), "Must be between 0 and 0.5.");

        if (EarlyStoppingMinDelta < 0f)
            throw new ArgumentOutOfRangeException(nameof(EarlyStoppingMinDelta), "Must be non-negative.");

        if (EarlyStoppingPatience < 0)
            throw new ArgumentOutOfRangeException(nameof(EarlyStoppingPatience), "Must be non-negative.");

        if (LrDecayPatience < 0)
            throw new ArgumentOutOfRangeException(nameof(LrDecayPatience), "Must be non-negative.");

        if (LrDecayFactor is <= 0f or >= 1f)
            throw new ArgumentOutOfRangeException(nameof(LrDecayFactor), "Must be between 0 and 1.");

        if (HiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(HiddenSize), "Must be positive.");
    }
}
