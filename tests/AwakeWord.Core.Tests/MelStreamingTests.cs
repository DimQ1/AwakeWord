using AwakeWord.Core;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AwakeWord.Core.Tests;

[TestClass]
public sealed class MelStreamingTests
{
    private static string ModelsDir => Path.Combine(
        Path.GetDirectoryName(typeof(MelStreamingTests).Assembly.Location)!,
        "..", "..", "..", "..", "..", "models");

    private static string TestDataDir => Path.Combine(
        Path.GetDirectoryName(typeof(MelStreamingTests).Assembly.Location)!,
        "TestData");

    [TestMethod]
    public void CompareMelOutputs_ChunkedVsWhole()
    {
        var melPath = Path.Combine(ModelsDir, "melspectrogram.onnx");
        if (!File.Exists(melPath)) { Assert.Inconclusive("Models not found"); return; }

        var audioFile = Path.Combine(TestDataDir, "hey_mycroft_test.wav");
        if (!File.Exists(audioFile)) { Assert.Inconclusive("Audio not found"); return; }

        var samples = AudioFileLoader.LoadAudioFile(audioFile);
        using var melSession = new InferenceSession(melPath);

        // 1. Process ENTIRE audio at once (ground truth)
        var wholeMel = RunMel(melSession, samples);
        Console.WriteLine($"Whole audio: {samples.Length} samples → {wholeMel.Length} mel frames");

        // 2. Process in 1280-sample chunks (independent, no overlap)
        var chunkedMel = new List<float[]>();
        for (var i = 0; i + 1280 <= samples.Length; i += 1280)
        {
            var chunk = new short[1280];
            Array.Copy(samples, i, chunk, 0, 1280);
            var frames = RunMel(melSession, chunk);
            chunkedMel.AddRange(frames);
        }
        Console.WriteLine($"Chunked independent: {chunkedMel.Count} mel frames");

        // 3. Growing-window approach (what the current code does)
        var growingMel = new List<float[]>();
        var growingWindow = new List<short>();
        var prevCount = 0;
        for (var i = 0; i + 1280 <= samples.Length; i += 1280)
        {
            for (var j = 0; j < 1280; j++)
                growingWindow.Add(samples[i + j]);
            var frames = RunMel(melSession, growingWindow.ToArray());
            for (var f = prevCount; f < frames.Length; f++)
                growingMel.Add(frames[f]);
            prevCount = frames.Length;
        }
        Console.WriteLine($"Growing window: {growingMel.Count} mel frames");

        // 4. Frame counts for various input sizes
        Console.WriteLine($"\n--- Mel frames for various input sizes ---");
        foreach (var size in new[] { 1280, 2560, 3840, 5120, 6400, 14080, 15232 })
        {
            if (size <= samples.Length)
            {
                var s = new short[size];
                Array.Copy(samples, 0, s, 0, size);
                var frames = RunMel(melSession, s);
                Console.WriteLine($"  {size} samples → {frames.Length} mel frames");
            }
        }

        // 5. Compare first 5 frames: chunked[0..4] vs whole[0..4] vs growing[0..4]
        Console.WriteLine($"\n--- Frame comparison (first 5) ---");
        for (var f = 0; f < Math.Min(5, wholeMel.Length); f++)
        {
            var wVals = string.Join(", ", wholeMel[f].Take(4).Select(v => v.ToString("F4")));
            var cVals = f < chunkedMel.Count ? string.Join(", ", chunkedMel[f].Take(4).Select(v => v.ToString("F4"))) : "N/A";
            var gVals = f < growingMel.Count ? string.Join(", ", growingMel[f].Take(4).Select(v => v.ToString("F4"))) : "N/A";
            Console.WriteLine($"  Frame {f}: whole=[{wVals}] chunk=[{cVals}] grow=[{gVals}]");
        }

        // 6. Compare growing vs whole
        var minLen = Math.Min(wholeMel.Length, growingMel.Count);
        var maxErr = 0.0;
        for (var f = 0; f < minLen; f++)
            for (var b = 0; b < 32; b++)
                maxErr = Math.Max(maxErr, Math.Abs(wholeMel[f][b] - growingMel[f][b]));
        Console.WriteLine($"\nGrowing vs Whole: max error = {maxErr:E4} over {minLen} frames");
    }

    private static float[][] RunMel(InferenceSession session, short[] pcm)
    {
        var floatPcm = new float[pcm.Length];
        for (var i = 0; i < pcm.Length; i++)
            floatPcm[i] = pcm[i];

        var inputTensor = new DenseTensor<float>(new[] { 1, pcm.Length });
        floatPcm.AsSpan().CopyTo(inputTensor.Buffer.Span);

        using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", inputTensor) });
        var rawOutput = results.First().AsTensor<float>();
        var dims = rawOutput.Dimensions.ToArray();

        var timeSteps = dims[2];
        var rows = new float[timeSteps][];
        for (var t = 0; t < timeSteps; t++)
        {
            rows[t] = new float[32];
            for (var b = 0; b < 32; b++)
                rows[t][b] = rawOutput[0, 0, t, b];
        }
        return rows;
    }
}
