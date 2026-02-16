using AwakeWord.Core;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AwakeWord.Training;

/// <summary>
/// Extracts fixed-size embedding feature windows from audio clips using the
/// openWakeWord mel-spectrogram and embedding models.
/// Each clip produces one or more [FeatureFrames, EmbeddingDim] feature windows
/// that can be used as training samples for a classifier.
/// </summary>
public sealed class EmbeddingExtractor : IDisposable
{
    private const int MelFrames = 76;
    private const int MelBins = 32;
    private const int MelStep = 8;
    private const int EmbeddingDim = 96;

    private readonly InferenceSession _melSession;
    private readonly InferenceSession _embeddingSession;
    private readonly int _featureFrames;

    public EmbeddingExtractor(string melModelPath, string embeddingModelPath, int featureFrames = 16)
    {
        _melSession = new InferenceSession(melModelPath);
        _embeddingSession = new InferenceSession(embeddingModelPath);
        _featureFrames = featureFrames;
    }

    /// <summary>
    /// Extract all embedding feature windows from an audio file.
    /// Returns a list of feature windows, each of shape [FeatureFrames, EmbeddingDim].
    /// </summary>
    public List<float[]> ExtractFeatureWindows(string audioFilePath)
    {
        var samples = AudioFileLoader.LoadAudioFile(audioFilePath);
        return ExtractFeatureWindows(samples);
    }

    /// <summary>
    /// Extract all embedding feature windows from raw PCM samples (16 kHz mono).
    /// Returns a list of flattened feature windows, each of size FeatureFrames * EmbeddingDim.
    /// </summary>
    public List<float[]> ExtractFeatureWindows(short[] pcmSamples)
    {
        var embeddings = ComputeAllEmbeddings(pcmSamples);
        var windows = new List<float[]>();

        // Slide a window of FeatureFrames over the embeddings
        for (var i = 0; i <= embeddings.Count - _featureFrames; i++)
        {
            var window = new float[_featureFrames * EmbeddingDim];
            for (var f = 0; f < _featureFrames; f++)
                Array.Copy(embeddings[i + f], 0, window, f * EmbeddingDim, EmbeddingDim);
            windows.Add(window);
        }

        return windows;
    }

    /// <summary>
    /// Compute all embeddings for a PCM clip using mel → embedding pipeline.
    /// </summary>
    private List<float[]> ComputeAllEmbeddings(short[] pcm)
    {
        var melFrames = ComputeMelspectrogram(pcm);
        var embeddings = new List<float[]>();

        for (var i = 0; i <= melFrames.Length - MelFrames; i += MelStep)
        {
            var window = new float[MelFrames * MelBins];
            for (var f = 0; f < MelFrames; f++)
                Array.Copy(melFrames[i + f], 0, window, f * MelBins, MelBins);

            var inputTensor = new DenseTensor<float>(new[] { 1, MelFrames, MelBins, 1 });
            window.AsSpan().CopyTo(inputTensor.Buffer.Span);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_1", inputTensor)
            };

            using var results = _embeddingSession.Run(inputs);
            var rawOutput = results.First().AsTensor<float>();

            var emb = new float[EmbeddingDim];
            for (var j = 0; j < EmbeddingDim; j++)
                emb[j] = rawOutput[0, 0, 0, j];
            embeddings.Add(emb);
        }

        return embeddings;
    }

    private float[][] ComputeMelspectrogram(short[] pcm)
    {
        var floatPcm = new float[pcm.Length];
        for (var i = 0; i < pcm.Length; i++)
            floatPcm[i] = pcm[i];

        var inputTensor = new DenseTensor<float>(new[] { 1, pcm.Length });
        floatPcm.AsSpan().CopyTo(inputTensor.Buffer.Span);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = _melSession.Run(inputs);
        var rawOutput = results.First().AsTensor<float>();
        var dims = rawOutput.Dimensions.ToArray();

        var timeSteps = dims[2];
        var rows = new float[timeSteps][];
        for (var t = 0; t < timeSteps; t++)
        {
            rows[t] = new float[MelBins];
            for (var b = 0; b < MelBins; b++)
                rows[t][b] = rawOutput[0, 0, t, b] / 10f + 2f;
        }

        return rows;
    }

    public void Dispose()
    {
        _melSession.Dispose();
        _embeddingSession.Dispose();
    }
}
