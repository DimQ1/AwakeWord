namespace AwakeWord.Training;

/// <summary>
/// Exports a trained <see cref="WakeWordClassifier"/> to ONNX format.
/// Generates a minimal ONNX protobuf file that matches the openWakeWord
/// classifier architecture:
///   Flatten → Gemm → LayerNorm → ReLU → Gemm → LayerNorm → ReLU → Gemm → LayerNorm → ReLU → Gemm → Sigmoid
/// Input shape: [1, FeatureFrames, EmbeddingDim]
/// Output shape: [1, 1]
/// </summary>
public static class OnnxModelExporter
{
    /// <summary>
    /// Export the trained classifier weights to an ONNX model file.
    /// The model accepts input [1, featureFrames, embeddingDim] and outputs [1, 1].
    /// </summary>
    public static void Export(ModelWeights weights, int featureFrames, int embeddingDim, string outputPath)
    {
        ValidateWeights(weights);

        var dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            Directory.CreateDirectory(dir);

        using var stream = File.Create(outputPath);
        WriteOnnxModel(stream, weights, featureFrames, embeddingDim);
    }

    private static void ValidateWeights(ModelWeights weights)
    {
        static void Check(float[] arr, string name)
        {
            for (var i = 0; i < arr.Length; i++)
            {
                if (float.IsNaN(arr[i]))
                    throw new InvalidOperationException($"Weight {name}[{i}] is NaN. Training may have diverged — try a lower learning rate.");
                if (float.IsInfinity(arr[i]))
                    throw new InvalidOperationException($"Weight {name}[{i}] is Infinity. Training may have diverged — try a lower learning rate.");
            }
        }

        Check(weights.W1, "W1");
        Check(weights.B1, "B1");
        Check(weights.Gamma1, "Gamma1");
        Check(weights.Beta1, "Beta1");
        Check(weights.W2, "W2");
        Check(weights.B2, "B2");
        Check(weights.Gamma2, "Gamma2");
        Check(weights.Beta2, "Beta2");
        Check(weights.W3, "W3");
        Check(weights.B3, "B3");
        Check(weights.Gamma3, "Gamma3");
        Check(weights.Beta3, "Beta3");
        Check(weights.W4, "W4");

        if (float.IsNaN(weights.B4))
            throw new InvalidOperationException("Weight B4 is NaN. Training may have diverged — try a lower learning rate.");
        if (float.IsInfinity(weights.B4))
            throw new InvalidOperationException("Weight B4 is Infinity. Training may have diverged — try a lower learning rate.");
    }

    /// <summary>
    /// Write a minimal ONNX protobuf model to the stream.
    /// We write raw protobuf bytes to avoid a dependency on Google.Protobuf / Onnx NuGet packages.
    /// 
    /// The ONNX graph structure:
    ///   input: "onnx::Flatten_0" [1, featureFrames, embeddingDim]
    ///   Flatten → "flatten_out"
    ///   Gemm(flatten_out, W1, B1, transB=1) → "dense1_out"
    ///   LayerNormalization(dense1_out, gamma1, beta1) → "ln1_out"
    ///   Relu → "relu1_out"
    ///   Gemm(relu1_out, W2, B2, transB=1) → "dense2_out"
    ///   LayerNormalization(dense2_out, gamma2, beta2) → "ln2_out"
    ///   Relu → "relu2_out"
    ///   Gemm(relu2_out, W3, B3, transB=1) → "dense3_out"
    ///   LayerNormalization(dense3_out, gamma3, beta3) → "ln3_out"
    ///   Relu → "relu3_out"
    ///   Gemm(relu3_out, W4, B4, transB=1) → "dense4_out"
    ///   Sigmoid → output [1, 1]
    /// </summary>
    private static void WriteOnnxModel(Stream stream, ModelWeights weights, int featureFrames, int embeddingDim)
    {
        var builder = new OnnxProtobufBuilder();

        var inputSize = weights.InputSize;
        var hiddenSize = weights.HiddenSize;

        // ── Initializers (constant tensors) ──

        // W1: [hiddenSize, inputSize] — stored as-is (Gemm with transB=1 does the transpose)
        builder.AddFloatTensor("W1", weights.W1, new[] { hiddenSize, inputSize });
        builder.AddFloatTensor("B1", weights.B1, new[] { hiddenSize });
        builder.AddFloatTensor("gamma1", weights.Gamma1, new[] { hiddenSize });
        builder.AddFloatTensor("beta1", weights.Beta1, new[] { hiddenSize });

        // W2: [hiddenSize, hiddenSize]
        builder.AddFloatTensor("W2", weights.W2, new[] { hiddenSize, hiddenSize });
        builder.AddFloatTensor("B2", weights.B2, new[] { hiddenSize });
        builder.AddFloatTensor("gamma2", weights.Gamma2, new[] { hiddenSize });
        builder.AddFloatTensor("beta2", weights.Beta2, new[] { hiddenSize });

        // W3: [hiddenSize, hiddenSize]
        builder.AddFloatTensor("W3", weights.W3, new[] { hiddenSize, hiddenSize });
        builder.AddFloatTensor("B3", weights.B3, new[] { hiddenSize });
        builder.AddFloatTensor("gamma3", weights.Gamma3, new[] { hiddenSize });
        builder.AddFloatTensor("beta3", weights.Beta3, new[] { hiddenSize });

        // W4: [1, hiddenSize]
        builder.AddFloatTensor("W4", weights.W4, new[] { 1, hiddenSize });
        builder.AddFloatTensor("B4", new[] { weights.B4 }, new[] { 1 });

        // ── Nodes ──
        var transB = new Dictionary<string, long> { { "transB", 1 } };
        var lnAttrs = new Dictionary<string, float> { { "epsilon", 1e-5f } };

        builder.AddNode("Flatten", new[] { "onnx::Flatten_0" }, new[] { "flatten_out" }, "Flatten",
            intAttrs: new Dictionary<string, long> { { "axis", 1 } });

        // Dense 1 → LayerNorm 1 → ReLU 1
        builder.AddNode("Gemm", new[] { "flatten_out", "W1", "B1" }, new[] { "dense1_out" }, "Gemm_0",
            intAttrs: transB);
        builder.AddNode("LayerNormalization", new[] { "dense1_out", "gamma1", "beta1" }, new[] { "ln1_out" }, "LayerNorm_0",
            floatAttrs: lnAttrs);
        builder.AddNode("Relu", new[] { "ln1_out" }, new[] { "relu1_out" }, "Relu_0");

        // Dense 2 → LayerNorm 2 → ReLU 2
        builder.AddNode("Gemm", new[] { "relu1_out", "W2", "B2" }, new[] { "dense2_out" }, "Gemm_1",
            intAttrs: transB);
        builder.AddNode("LayerNormalization", new[] { "dense2_out", "gamma2", "beta2" }, new[] { "ln2_out" }, "LayerNorm_1",
            floatAttrs: lnAttrs);
        builder.AddNode("Relu", new[] { "ln2_out" }, new[] { "relu2_out" }, "Relu_1");

        // Dense 3 → LayerNorm 3 → ReLU 3
        builder.AddNode("Gemm", new[] { "relu2_out", "W3", "B3" }, new[] { "dense3_out" }, "Gemm_2",
            intAttrs: transB);
        builder.AddNode("LayerNormalization", new[] { "dense3_out", "gamma3", "beta3" }, new[] { "ln3_out" }, "LayerNorm_2",
            floatAttrs: lnAttrs);
        builder.AddNode("Relu", new[] { "ln3_out" }, new[] { "relu3_out" }, "Relu_2");

        // Dense 4 → Sigmoid
        builder.AddNode("Gemm", new[] { "relu3_out", "W4", "B4" }, new[] { "dense4_out" }, "Gemm_3",
            intAttrs: transB);
        builder.AddNode("Sigmoid", new[] { "dense4_out" }, new[] { "output" }, "Sigmoid_0");

        // ── Input / Output ──
        builder.SetInput("onnx::Flatten_0", new[] { 1, featureFrames, embeddingDim });
        builder.SetOutput("output", new[] { 1, 1 });

        builder.WriteTo(stream);
    }
}

/// <summary>
/// Minimal ONNX protobuf builder that constructs a valid .onnx file without
/// requiring the Google.Protobuf or OnnxSharp NuGet packages.
/// Supports the subset of ONNX needed for a simple classifier graph.
/// </summary>
internal sealed class OnnxProtobufBuilder
{
    // Protobuf field identifiers for ONNX structures
    // See https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

    private readonly List<byte[]> _initializers = new();
    private readonly List<byte[]> _nodes = new();
    private byte[]? _input;
    private byte[]? _output;

    public void AddFloatTensor(string name, float[] data, int[] shape)
    {
        using var ms = new MemoryStream();
        // TensorProto – fields written in field-number order for canonical encoding
        // dims: repeated int64, field 1 (unpacked per ONNX proto2 definition)
        foreach (var dim in shape)
            WriteField(ms, 1, (long)dim);
        WriteField(ms, 2, 1L);  // data_type = FLOAT (1), field 2
        WriteField(ms, 8, name); // name, field 8
        // raw_data (bytes), field 9 — stores floats as raw little-endian bytes
        // This matches the encoding used by all reference ONNX models and is
        // required for compatibility with Netron and other ONNX visualization tools.
        WriteRawFloatDataField(ms, 9, data);

        _initializers.Add(ms.ToArray());
    }

    public void AddInt64Tensor(string name, long[] data, int[] shape)
    {
        using var ms = new MemoryStream();
        // dims: repeated int64, field 1 (unpacked per ONNX proto2 definition)
        foreach (var dim in shape)
            WriteField(ms, 1, (long)dim);
        WriteField(ms, 2, 7L);  // data_type = INT64 (7)
        WriteField(ms, 8, name); // name, field 8
        // raw_data (bytes), field 9 — stores int64s as raw little-endian bytes
        WriteRawInt64DataField(ms, 9, data);

        _initializers.Add(ms.ToArray());
    }

    public void AddNode(string opType, string[] inputs, string[] outputs, string name,
        Dictionary<string, long>? intAttrs = null,
        Dictionary<string, float>? floatAttrs = null)
    {
        using var ms = new MemoryStream();
        foreach (var inp in inputs)
            WriteField(ms, 1, inp);   // input (repeated string), field 1
        foreach (var outp in outputs)
            WriteField(ms, 2, outp);  // output (repeated string), field 2
        WriteField(ms, 3, name);      // name, field 3
        WriteField(ms, 4, opType);    // op_type, field 4

        if (intAttrs != null)
        {
            foreach (var (attrName, attrValue) in intAttrs)
            {
                // AttributeProto: name (field 1), type (field 20, INT=2), i (field 3)
                using var attrMs = new MemoryStream();
                WriteField(attrMs, 1, attrName);   // name
                WriteField(attrMs, 3, attrValue);   // i (int value)
                WriteField(attrMs, 20, 2L);         // type = INT
                WriteSubmessageField(ms, 5, attrMs.ToArray()); // attribute, field 5
            }
        }

        if (floatAttrs != null)
        {
            foreach (var (attrName, attrValue) in floatAttrs)
            {
                // AttributeProto: name (field 1), f (field 2, float), type (field 20, FLOAT=1)
                using var attrMs = new MemoryStream();
                WriteField(attrMs, 1, attrName);
                WriteFixedFloatField(attrMs, 2, attrValue);
                WriteField(attrMs, 20, 1L);         // type = FLOAT
                WriteSubmessageField(ms, 5, attrMs.ToArray());
            }
        }

        _nodes.Add(ms.ToArray());
    }

    public void SetInput(string name, int[] shape)
    {
        _input = BuildValueInfo(name, shape);
    }

    public void SetOutput(string name, int[] shape)
    {
        _output = BuildValueInfo(name, shape);
    }

    public void WriteTo(Stream output)
    {
        // Build GraphProto
        using var graphMs = new MemoryStream();

        // nodes (repeated NodeProto), field 1
        foreach (var node in _nodes)
            WriteSubmessageField(graphMs, 1, node);

        // name (string), field 2
        WriteField(graphMs, 2, "wake_word_classifier");

        // initializer (repeated TensorProto), field 5
        foreach (var init in _initializers)
            WriteSubmessageField(graphMs, 5, init);

        // input (repeated ValueInfoProto), field 11
        if (_input != null)
            WriteSubmessageField(graphMs, 11, _input);

        // output (repeated ValueInfoProto), field 12
        if (_output != null)
            WriteSubmessageField(graphMs, 12, _output);

        var graphBytes = graphMs.ToArray();

        // Build ModelProto
        using var modelMs = new MemoryStream();
        WriteField(modelMs, 1, 8L);           // ir_version = 8, field 1
        WriteField(modelMs, 2, "awakeword-trainer"); // producer_name, field 2
        WriteField(modelMs, 3, "1.0");                // producer_version, field 3
        WriteField(modelMs, 5, 1L);                   // model_version, field 5

        WriteSubmessageField(modelMs, 7, graphBytes);  // graph, field 7

        // opset_import (OperatorSetIdProto), field 8
        using var opsetMs = new MemoryStream();
        WriteField(opsetMs, 1, "");            // domain (default)
        WriteField(opsetMs, 2, 17L);           // version = 17
        WriteSubmessageField(modelMs, 8, opsetMs.ToArray());

        modelMs.WriteTo(output);
        output.Flush();
    }

    private static byte[] BuildValueInfo(string name, int[] shape)
    {
        // ValueInfoProto: name (field 1), type (TypeProto, field 2)
        using var viMs = new MemoryStream();
        WriteField(viMs, 1, name);

        // TypeProto → tensor_type (field 1) → TensorTypeProto
        using var tensorTypeMs = new MemoryStream();
        WriteField(tensorTypeMs, 1, 1L); // elem_type = FLOAT (1), field 1

        // TensorShapeProto (field 2)
        using var shapeMs = new MemoryStream();
        foreach (var dim in shape)
        {
            using var dimMs = new MemoryStream();
            WriteField(dimMs, 1, (long)dim); // dim_value (int64), field 1
            WriteSubmessageField(shapeMs, 1, dimMs.ToArray()); // dim (repeated), field 1
        }
        WriteSubmessageField(tensorTypeMs, 2, shapeMs.ToArray()); // shape, field 2

        // TypeProto: tensor_type, field 1
        using var typeMs = new MemoryStream();
        WriteSubmessageField(typeMs, 1, tensorTypeMs.ToArray());

        WriteSubmessageField(viMs, 2, typeMs.ToArray());

        return viMs.ToArray();
    }

    // ── Protobuf encoding helpers ──

    private static void WriteField(Stream s, int fieldNumber, string value)
    {
        // Wire type 2 (length-delimited)
        WriteVarint(s, (uint)(fieldNumber << 3 | 2));
        var bytes = System.Text.Encoding.UTF8.GetBytes(value);
        WriteVarint(s, (uint)bytes.Length);
        s.Write(bytes, 0, bytes.Length);
    }

    private static void WriteField(Stream s, int fieldNumber, long value)
    {
        // Wire type 0 (varint)
        WriteVarint(s, (uint)(fieldNumber << 3 | 0));
        WriteVarint(s, (ulong)value);
    }

    private static void WriteSubmessageField(Stream s, int fieldNumber, byte[] data)
    {
        // Wire type 2 (length-delimited)
        WriteVarint(s, (uint)(fieldNumber << 3 | 2));
        WriteVarint(s, (uint)data.Length);
        s.Write(data, 0, data.Length);
    }

    private static void WriteRawFloatDataField(Stream s, int fieldNumber, float[] data)
    {
        // Wire type 2 (length-delimited) with raw little-endian float bytes
        WriteVarint(s, (uint)(fieldNumber << 3 | 2));
        WriteVarint(s, (uint)(data.Length * 4));
        foreach (var f in data)
        {
            var bytes = BitConverter.GetBytes(f);
            s.Write(bytes, 0, 4);
        }
    }

    private static void WriteFixedFloatField(Stream s, int fieldNumber, float value)
    {
        // Wire type 5 = 32-bit (fixed32) for float
        WriteVarint(s, (uint)(fieldNumber << 3 | 5));
        var bytes = BitConverter.GetBytes(value);
        s.Write(bytes, 0, 4);
    }

    private static void WriteRawInt64DataField(Stream s, int fieldNumber, long[] data)
    {
        // Wire type 2 (length-delimited) with raw little-endian int64 bytes
        WriteVarint(s, (uint)(fieldNumber << 3 | 2));
        WriteVarint(s, (uint)(data.Length * 8));
        foreach (var v in data)
        {
            var bytes = BitConverter.GetBytes(v);
            s.Write(bytes, 0, 8);
        }
    }

    private static void WriteVarint(Stream s, ulong value)
    {
        do
        {
            var b = (byte)(value & 0x7F);
            value >>= 7;
            if (value > 0) b |= 0x80;
            s.WriteByte(b);
        } while (value > 0);
    }
}
