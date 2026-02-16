using AwakeWord.Training;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AwakeWord.Core.Tests;

[TestClass]
public class OnnxExportTests
{
    private const int FeatureFrames = 16;
    private const int EmbeddingDim = 96;
    private const int HiddenSize = 128;
    private const int InputSize = FeatureFrames * EmbeddingDim; // 1536

    [TestMethod]
    public void Export_KnownWeights_ProducesValidModel()
    {
        // Arrange: create known weights (small values, no NaN/Inf)
        var weights = CreateTestWeights(scale: 0.01f);
        var modelPath = Path.Combine(Path.GetTempPath(), $"test_export_{Guid.NewGuid()}.onnx");

        try
        {
            // Act: export
            OnnxModelExporter.Export(weights, FeatureFrames, EmbeddingDim, modelPath);

            // Assert: file exists and is non-empty
            Assert.IsTrue(File.Exists(modelPath), "ONNX model file was not created");
            var fileSize = new FileInfo(modelPath).Length;
            Assert.IsTrue(fileSize > 0, "ONNX model file is empty");

            // Assert: file size should be approximately 920 KB for 230K+ parameter model
            // W1: 128*1536*4=786432, W2: 128*128*4=65536, W3: 128*128*4=65536, other: ~4096
            var expectedMinSize = 786432 + 65536 + 65536 + 4096; // ~917 KB minimum
            Console.WriteLine($"Model file size: {fileSize} bytes ({fileSize / 1024.0:F1} KB)");
            Assert.IsTrue(fileSize >= expectedMinSize,
                $"Model file too small: {fileSize} bytes, expected at least {expectedMinSize}");

            // Assert: file starts with valid protobuf (field 1 = ir_version, wire type 0)
            var header = File.ReadAllBytes(modelPath).Take(2).ToArray();
            Assert.AreEqual(0x08, header[0], $"First byte should be 0x08 (field 1, varint), got 0x{header[0]:X2}");

            // Load with OnnxRuntime
            using var session = new InferenceSession(modelPath);
            Assert.AreEqual(1, session.InputMetadata.Count, "Expected exactly 1 input");

            var inputName = session.InputMetadata.Keys.First();
            Assert.AreEqual("onnx::Flatten_0", inputName);

            // Verify output metadata
            Assert.AreEqual(1, session.OutputMetadata.Count, "Expected exactly 1 output");
        }
        finally
        {
            if (File.Exists(modelPath)) File.Delete(modelPath);
        }
    }

    [TestMethod]
    public void Export_KnownWeights_InferenceReturnsFiniteValue()
    {
        // Arrange
        var weights = CreateTestWeights(scale: 0.01f);
        var modelPath = Path.Combine(Path.GetTempPath(), $"test_inference_{Guid.NewGuid()}.onnx");

        try
        {
            OnnxModelExporter.Export(weights, FeatureFrames, EmbeddingDim, modelPath);

            using var session = new InferenceSession(modelPath);
            var inputName = session.InputMetadata.Keys.First();

            // Create test input: small random values (avoid all-zeros which is a
            // degenerate case for LayerNorm — zero variance on constant input)
            var rng = new Random(99);
            var input = new DenseTensor<float>(new[] { 1, FeatureFrames, EmbeddingDim });
            for (var f = 0; f < FeatureFrames; f++)
                for (var d = 0; d < EmbeddingDim; d++)
                    input[0, f, d] = (float)(rng.NextDouble() * 0.01);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, input)
            };

            using var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();
            var value = output[0, 0];

            Console.WriteLine($"Zero input → output: {value}");
            Assert.IsFalse(float.IsNaN(value), $"Output is NaN for zero input!");
            Assert.IsFalse(float.IsInfinity(value), $"Output is Infinity for zero input!");
            Assert.IsTrue(value >= 0f && value <= 1f, $"Output {value} is outside [0, 1] range");
        }
        finally
        {
            if (File.Exists(modelPath)) File.Delete(modelPath);
        }
    }

    [TestMethod]
    public void Export_KnownWeights_InferenceMatchesCSharpClassifier()
    {
        // Arrange: compare C# classifier output vs ONNX model output
        var classifier = new WakeWordClassifier(FeatureFrames, EmbeddingDim, HiddenSize);
        var weights = classifier.GetWeights();
        var modelPath = Path.Combine(Path.GetTempPath(), $"test_match_{Guid.NewGuid()}.onnx");

        try
        {
            OnnxModelExporter.Export(weights, FeatureFrames, EmbeddingDim, modelPath);

            // Create a test input with known values
            var rng = new Random(123);
            var testInput = new float[InputSize];
            for (var i = 0; i < InputSize; i++)
                testInput[i] = (float)(rng.NextDouble() * 2 - 1);

            // C# forward pass
            var csharpOutput = classifier.Forward(testInput);
            Console.WriteLine($"C# classifier output: {csharpOutput}");

            // ONNX forward pass
            using var session = new InferenceSession(modelPath);
            var inputName = session.InputMetadata.Keys.First();
            var inputTensor = new DenseTensor<float>(new[] { 1, FeatureFrames, EmbeddingDim });

            // Copy test input to 3D tensor
            for (var f = 0; f < FeatureFrames; f++)
                for (var d = 0; d < EmbeddingDim; d++)
                    inputTensor[0, f, d] = testInput[f * EmbeddingDim + d];

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            using var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();
            var onnxOutput = output[0, 0];
            Console.WriteLine($"ONNX model output: {onnxOutput}");

            Assert.IsFalse(float.IsNaN(onnxOutput), $"ONNX output is NaN!");
            Assert.IsFalse(float.IsInfinity(onnxOutput), $"ONNX output is Infinity!");

            // They should match closely
            var diff = MathF.Abs(csharpOutput - onnxOutput);
            Console.WriteLine($"Difference: {diff}");
            Assert.IsTrue(diff < 0.001f,
                $"C# output ({csharpOutput}) and ONNX output ({onnxOutput}) differ by {diff}");
        }
        finally
        {
            if (File.Exists(modelPath)) File.Delete(modelPath);
        }
    }

    [TestMethod]
    public void Export_TrainedWeights_InferenceReturnsFiniteValue()
    {
        // Simulate a training scenario with a few epochs
        var classifier = new WakeWordClassifier(FeatureFrames, EmbeddingDim, HiddenSize);
        var rng = new Random(42);

        // Create some fake training data
        for (var epoch = 0; epoch < 10; epoch++)
        {
            // Positive sample
            var pos = new float[InputSize];
            for (var i = 0; i < InputSize; i++)
                pos[i] = (float)(rng.NextDouble() * 2);  // slightly biased positive
            var loss = classifier.TrainStep(pos, 1f, 0.001f, 1f, 1f, 0f);
            Console.WriteLine($"Epoch {epoch} pos loss: {loss}");

            // Negative sample
            var neg = new float[InputSize];
            for (var i = 0; i < InputSize; i++)
                neg[i] = (float)(rng.NextDouble() * 2 - 2); // slightly biased negative
            loss = classifier.TrainStep(neg, 0f, 0.001f, 1f, 1f, 0f);
            Console.WriteLine($"Epoch {epoch} neg loss: {loss}");
        }

        var weights = classifier.GetWeights();

        // Check weights for NaN/Inf
        Assert.IsFalse(weights.W1.Any(float.IsNaN), "W1 contains NaN after training");
        Assert.IsFalse(weights.W1.Any(float.IsInfinity), "W1 contains Infinity after training");
        Assert.IsFalse(weights.B1.Any(float.IsNaN), "B1 contains NaN after training");
        Assert.IsFalse(weights.Gamma1.Any(float.IsNaN), "Gamma1 contains NaN after training");
        Assert.IsFalse(weights.Beta1.Any(float.IsNaN), "Beta1 contains NaN after training");
        Assert.IsFalse(weights.W2.Any(float.IsNaN), "W2 contains NaN after training");
        Assert.IsFalse(weights.B2.Any(float.IsNaN), "B2 contains NaN after training");
        Assert.IsFalse(weights.Gamma2.Any(float.IsNaN), "Gamma2 contains NaN after training");
        Assert.IsFalse(weights.Beta2.Any(float.IsNaN), "Beta2 contains NaN after training");
        Assert.IsFalse(weights.W3.Any(float.IsNaN), "W3 contains NaN after training");
        Assert.IsFalse(weights.B3.Any(float.IsNaN), "B3 contains NaN after training");
        Assert.IsFalse(weights.Gamma3.Any(float.IsNaN), "Gamma3 contains NaN after training");
        Assert.IsFalse(weights.Beta3.Any(float.IsNaN), "Beta3 contains NaN after training");
        Assert.IsFalse(weights.W4.Any(float.IsNaN), "W4 contains NaN after training");
        Assert.IsFalse(float.IsNaN(weights.B4), "B4 is NaN after training");

        var modelPath = Path.Combine(Path.GetTempPath(), $"test_trained_{Guid.NewGuid()}.onnx");

        try
        {
            OnnxModelExporter.Export(weights, FeatureFrames, EmbeddingDim, modelPath);

            using var session = new InferenceSession(modelPath);
            var inputName = session.InputMetadata.Keys.First();

            var input = new DenseTensor<float>(new[] { 1, FeatureFrames, EmbeddingDim });
            var testData = new float[InputSize];
            for (var i = 0; i < InputSize; i++)
                testData[i] = (float)(rng.NextDouble() * 2 - 1);
            for (var f = 0; f < FeatureFrames; f++)
                for (var d = 0; d < EmbeddingDim; d++)
                    input[0, f, d] = testData[f * EmbeddingDim + d];

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, input)
            };

            using var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();
            var value = output[0, 0];
            Console.WriteLine($"Trained model ONNX output: {value}");

            Assert.IsFalse(float.IsNaN(value), $"Trained model output is NaN!");
            Assert.IsFalse(float.IsInfinity(value), $"Trained model output is Infinity!");
            Assert.IsTrue(value >= 0f && value <= 1f, $"Output {value} is outside [0, 1]");
        }
        finally
        {
            if (File.Exists(modelPath)) File.Delete(modelPath);
        }
    }

    [TestMethod]
    public void Export_ProtobufStructure_IsWellFormed()
    {
        // This test validates the raw protobuf structure to ensure compatibility
        // with tools like Netron: every length-delimited field's declared length
        // must be fully backed by data in the file (no "Unexpected end of file").
        var weights = CreateTestWeights(scale: 0.01f);
        var modelPath = Path.Combine(Path.GetTempPath(), $"test_protobuf_{Guid.NewGuid()}.onnx");

        try
        {
            OnnxModelExporter.Export(weights, FeatureFrames, EmbeddingDim, modelPath);

            var bytes = File.ReadAllBytes(modelPath);
            Console.WriteLine($"Model file: {bytes.Length} bytes");

            // Walk the top-level protobuf fields (ModelProto) and verify all
            // length-delimited fields are fully contained within the file.
            var pos = 0;
            while (pos < bytes.Length)
            {
                var (tag, newPos) = ReadVarint(bytes, pos);
                pos = newPos;

                var fieldNumber = tag >> 3;
                var wireType = tag & 0x7;

                switch (wireType)
                {
                    case 0: // varint
                        var (_, nextPos) = ReadVarint(bytes, pos);
                        pos = nextPos;
                        break;
                    case 2: // length-delimited
                        var (length, afterLen) = ReadVarint(bytes, pos);
                        pos = afterLen;
                        Assert.IsTrue(pos + (int)length <= bytes.Length,
                            $"Field {fieldNumber}: declared length {length} at offset {pos} " +
                            $"exceeds file size {bytes.Length} (would need {pos + (int)length} bytes)");
                        pos += (int)length;
                        break;
                    case 5: // fixed32
                        Assert.IsTrue(pos + 4 <= bytes.Length,
                            $"Field {fieldNumber}: fixed32 at offset {pos} exceeds file size");
                        pos += 4;
                        break;
                    case 1: // fixed64
                        Assert.IsTrue(pos + 8 <= bytes.Length,
                            $"Field {fieldNumber}: fixed64 at offset {pos} exceeds file size");
                        pos += 8;
                        break;
                    default:
                        Assert.Fail($"Unknown wire type {wireType} for field {fieldNumber} at offset {pos}");
                        break;
                }
            }

            Assert.AreEqual(bytes.Length, pos, "Not all bytes consumed — trailing garbage");
            Console.WriteLine("Protobuf structure is well-formed");

            // Also verify the model can be reloaded from the file
            using var session = new InferenceSession(modelPath);
            Console.WriteLine($"OnnxRuntime loaded successfully: {session.InputMetadata.Count} inputs, {session.OutputMetadata.Count} outputs");
        }
        finally
        {
            if (File.Exists(modelPath)) File.Delete(modelPath);
        }
    }

    private static (ulong value, int newPos) ReadVarint(byte[] data, int pos)
    {
        ulong result = 0;
        var shift = 0;
        while (pos < data.Length)
        {
            var b = data[pos++];
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
        }
        return (result, pos);
    }

    private static ModelWeights CreateTestWeights(float scale)
    {
        var rng = new Random(42);

        var w1 = new float[HiddenSize * InputSize];
        for (var i = 0; i < w1.Length; i++)
            w1[i] = (float)(rng.NextDouble() * 2 - 1) * scale;

        var b1 = new float[HiddenSize];

        var gamma1 = new float[HiddenSize];
        Array.Fill(gamma1, 1f);
        var beta1 = new float[HiddenSize];

        var w2 = new float[HiddenSize * HiddenSize];
        for (var i = 0; i < w2.Length; i++)
            w2[i] = (float)(rng.NextDouble() * 2 - 1) * scale;

        var b2 = new float[HiddenSize];

        var gamma2 = new float[HiddenSize];
        Array.Fill(gamma2, 1f);
        var beta2 = new float[HiddenSize];

        var w3 = new float[HiddenSize * HiddenSize];
        for (var i = 0; i < w3.Length; i++)
            w3[i] = (float)(rng.NextDouble() * 2 - 1) * scale;

        var b3 = new float[HiddenSize];

        var gamma3 = new float[HiddenSize];
        Array.Fill(gamma3, 1f);
        var beta3 = new float[HiddenSize];

        var w4 = new float[HiddenSize];
        for (var i = 0; i < HiddenSize; i++)
            w4[i] = (float)(rng.NextDouble() * 2 - 1) * scale;

        return new ModelWeights
        {
            InputSize = InputSize,
            HiddenSize = HiddenSize,
            W1 = w1,
            B1 = b1,
            Gamma1 = gamma1,
            Beta1 = beta1,
            W2 = w2,
            B2 = b2,
            Gamma2 = gamma2,
            Beta2 = beta2,
            W3 = w3,
            B3 = b3,
            Gamma3 = gamma3,
            Beta3 = beta3,
            W4 = w4,
            B4 = 0f
        };
    }
}
