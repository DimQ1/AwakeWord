using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AwakeWord.Core;

/// <summary>
/// Implements the full openWakeWord pipeline in C#:
///   raw 16-bit PCM → mel-spectrogram → embedding → wake-word classifier.
/// Processes audio in a streaming fashion (1280-sample / 80 ms chunks).
/// </summary>
public sealed class OnnxWakeWordDetector : IAwakeWordDetector
{
    // ── Pipeline constants (matching openWakeWord Python implementation) ──
    private const int SampleRate = 16000;
    private const int ChunkSize = 1280;           // 80 ms @ 16 kHz
    private const int MelFrames = 76;              // frames per embedding window
    private const int MelBins = 32;
    private const int MelStep = 8;                 // frames to step between embedding windows (batch mode)
    private const int EmbeddingDim = 96;

    // ── ONNX sessions ──
    private readonly InferenceSession _melspecSession;
    private readonly InferenceSession _embeddingSession;
    private readonly InferenceSession _wakeWordSession;
    private readonly string _wakeWordInputName;

    private readonly WakeWordConfig _config;

    // ── Streaming buffers ──
    // PCM sample buffer – accumulates until we have a full ChunkSize block
    private readonly List<short> _pcmBuffer = new();

    // Sliding window of raw PCM for mel computation (to maintain STFT overlap context)
    private readonly List<short> _melPcmWindow = new();
    private int _melFramesProduced;

    // mel-spectrogram ring buffer (rows = up to ~10 s worth of frames)
    private readonly List<float[]> _melBuffer = new();
    private const int MelMaxLen = 10 * 97;

    // embedding feature ring buffer (rows = embedding vectors of size 96)
    private readonly List<float[]> _featureBuffer = new();
    private const int FeatureMaxLen = 120;

    public OnnxWakeWordDetector(WakeWordConfig config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _config.Validate();

        var opts = new SessionOptions();
        opts.InterOpNumThreads = _config.NumThreads;
        opts.IntraOpNumThreads = _config.NumThreads;

        _melspecSession = new InferenceSession(_config.MelSpectrogramModelPath, opts);
        _embeddingSession = new InferenceSession(_config.EmbeddingModelPath, opts);
        _wakeWordSession = new InferenceSession(_config.WakeWordModelPath, opts);

        // Read the wake-word classifier input tensor name from model metadata
        _wakeWordInputName = _wakeWordSession.InputMetadata.Keys.First();

        // Initialise mel buffer with zeros (matches Python's np.zeros((76, 32)))
        for (var i = 0; i < MelFrames; i++)
        {
            var row = new float[MelBins];
            _melBuffer.Add(row);
        }

        WarmUpFeatureBuffer();
    }

    public string WakeWord => _config.WakeWord;

    // ── Public API ──────────────────────────────────────────────────────────

    public bool ProcessAudio(ReadOnlySpan<short> audioSamples, out float confidence)
    {
        confidence = 0f;

        // Buffer incoming samples
        foreach (var s in audioSamples)
            _pcmBuffer.Add(s);

        var anyProcessed = false;

        // Process in ChunkSize (1280-sample) blocks.
        while (_pcmBuffer.Count >= ChunkSize)
        {
            // Move ChunkSize samples from pcm buffer to mel window
            for (var i = 0; i < ChunkSize; i++)
                _melPcmWindow.Add(_pcmBuffer[i]);
            _pcmBuffer.RemoveRange(0, ChunkSize);

            // 1. Compute mel-spectrogram over the entire mel window.
            //    The STFT inside the mel model needs overlap context from
            //    previous chunks to produce correct frame counts.
            var allMelFrames = ComputeMelspectrogram(_melPcmWindow.ToArray());

            // Only append NEW mel frames (ones not produced in previous calls)
            var numNewFrames = allMelFrames.Length - _melFramesProduced;
            for (var f = _melFramesProduced; f < allMelFrames.Length; f++)
                _melBuffer.Add(allMelFrames[f]);
            _melFramesProduced = allMelFrames.Length;

            // Trim mel PCM window to prevent unbounded growth.
            // Keep enough samples for the last 76 mel frames worth of context.
            const int maxPcmWindowSamples = ChunkSize * 20; // ~1.6 seconds
            if (_melPcmWindow.Count > maxPcmWindowSamples)
            {
                var trimCount = _melPcmWindow.Count - maxPcmWindowSamples;
                _melPcmWindow.RemoveRange(0, trimCount);
                // Recount: run mel on trimmed window to find current frame count
                var trimmedMel = ComputeMelspectrogram(_melPcmWindow.ToArray());
                _melFramesProduced = trimmedMel.Length;
            }

            // Trim mel buffer to prevent unbounded growth
            while (_melBuffer.Count > MelMaxLen)
                _melBuffer.RemoveAt(0);

            // 2. Compute ONE embedding per chunk from the last 76 mel frames.
            //    This matches Python openWakeWord which computes one embedding
            //    per 1280-sample chunk, not one per mel frame.
            if (_melBuffer.Count >= MelFrames)
            {
                var endIdx = _melBuffer.Count;
                var startIdx = endIdx - MelFrames;
                var embedding = ComputeEmbedding(startIdx, endIdx);
                _featureBuffer.Add(embedding);
            }

            TrimFeatureBuffer();
            anyProcessed = true;
        }

        // 3. Run wake-word classifier on the latest feature window
        if (anyProcessed && _featureBuffer.Count >= _config.FeatureFrames)
        {
            confidence = RunWakeWordClassifier();
            if (confidence >= _config.DetectionThreshold)
            {
                Reset();
                return true;
            }
        }

        return false;
    }

    public void Reset()
    {
        _pcmBuffer.Clear();
        _melPcmWindow.Clear();
        _melFramesProduced = 0;

        _melBuffer.Clear();
        for (var i = 0; i < MelFrames; i++)
        {
            var row = new float[MelBins];
            _melBuffer.Add(row);
        }

        _featureBuffer.Clear();
        WarmUpFeatureBuffer();
    }

    public void Dispose()
    {
        _melspecSession.Dispose();
        _embeddingSession.Dispose();
        _wakeWordSession.Dispose();
    }

    // ── Internals ───────────────────────────────────────────────────────────

    /// <summary>
    /// Fill the feature buffer with a random-noise embedding so the classifier
    /// has something to work with before real audio arrives (matches Python init).
    /// </summary>
    private void WarmUpFeatureBuffer()
    {
        var rng = new Random(42);
        var warmupSamples = new short[SampleRate * 4];
        for (var i = 0; i < warmupSamples.Length; i++)
            warmupSamples[i] = (short)rng.Next(-1000, 1000);

        var embeddings = ComputeEmbeddingsForClip(warmupSamples);
        foreach (var emb in embeddings)
            _featureBuffer.Add(emb);
    }

    /// <summary>Compute mel-spectrogram for a batch of raw PCM samples.</summary>
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

        using var results = _melspecSession.Run(inputs);
        var output = results.First();
        var rawOutput = output.AsTensor<float>();
        var dims = rawOutput.Dimensions.ToArray();

        // Output shape: [batch=1, 1, time, 32] – squeeze to [time, 32]
        var timeSteps = dims[2];
        var rows = new float[timeSteps][];
        for (var t = 0; t < timeSteps; t++)
        {
            rows[t] = new float[MelBins];
            for (var b = 0; b < MelBins; b++)
            {
                var val = rawOutput[0, 0, t, b];
                rows[t][b] = val / 10f + 2f; // transform matching Python
            }
        }

        return rows;
    }

    /// <summary>Compute a single 96-d embedding from a 76-frame mel window.</summary>
    private float[] ComputeEmbedding(int melStart, int melEnd)
    {
        var window = new float[MelFrames * MelBins];
        for (var f = 0; f < MelFrames; f++)
        {
            var row = _melBuffer[melStart + f];
            Array.Copy(row, 0, window, f * MelBins, MelBins);
        }

        // Shape: [1, 76, 32, 1]
        var inputTensor = new DenseTensor<float>(new[] { 1, MelFrames, MelBins, 1 });
        window.AsSpan().CopyTo(inputTensor.Buffer.Span);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_1", inputTensor)
        };

        using var results = _embeddingSession.Run(inputs);
        var output = results.First();
        var rawOutput = output.AsTensor<float>();

        // Output: [1, 1, 1, 96] → squeeze to [96]
        var embedding = new float[EmbeddingDim];
        for (var i = 0; i < EmbeddingDim; i++)
            embedding[i] = rawOutput[0, 0, 0, i];

        return embedding;
    }

    /// <summary>Compute all embeddings for a full audio clip (used for warm-up).</summary>
    private List<float[]> ComputeEmbeddingsForClip(short[] pcm)
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
            var output = results.First();
            var rawOutput = output.AsTensor<float>();

            var emb = new float[EmbeddingDim];
            for (var j = 0; j < EmbeddingDim; j++)
                emb[j] = rawOutput[0, 0, 0, j];

            embeddings.Add(emb);
        }

        return embeddings;
    }

    /// <summary>Run the wake-word classifier on the latest feature window.</summary>
    private float RunWakeWordClassifier()
    {
        var frames = _config.FeatureFrames;
        var startIdx = _featureBuffer.Count - frames;

        // Shape: [1, 16, 96]
        var inputTensor = new DenseTensor<float>(new[] { 1, frames, EmbeddingDim });
        for (var f = 0; f < frames; f++)
        {
            var row = _featureBuffer[startIdx + f];
            for (var d = 0; d < EmbeddingDim; d++)
                inputTensor[0, f, d] = row[d];
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_wakeWordInputName, inputTensor)
        };

        using var results = _wakeWordSession.Run(inputs);
        var output = results.First();
        var rawOutput = output.AsTensor<float>();

        return rawOutput[0, 0];
    }

    private void TrimFeatureBuffer()
    {
        while (_featureBuffer.Count > FeatureMaxLen)
            _featureBuffer.RemoveAt(0);
    }
}
