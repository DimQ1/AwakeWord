namespace AwakeWord.Training;

/// <summary>
/// A neural-network classifier trained via gradient descent.
/// Architecture:
///   Dense(input→hidden) → LayerNorm → ReLU →
///   Dense(hidden→hidden) → LayerNorm → ReLU →
///   Dense(hidden→hidden) → LayerNorm → ReLU →
///   Dense(hidden→1) → Sigmoid
/// Input shape: [FeatureFrames * EmbeddingDim] flattened.
/// Output: scalar probability [0..1].
/// </summary>
public sealed class WakeWordClassifier
{
    private readonly int _inputSize;   // FeatureFrames * EmbeddingDim (e.g. 16*96 = 1536)
    private readonly int _hiddenSize;  // hidden layer width (128 to match reference)
    private const float LayerNormEps = 1e-5f;

    // Layer 1: Dense (inputSize → hiddenSize)
    private readonly float[] _w1;       // [hiddenSize, inputSize]
    private readonly float[] _b1;       // [hiddenSize]

    // LayerNorm 1
    private readonly float[] _gamma1;   // [hiddenSize]
    private readonly float[] _beta1;    // [hiddenSize]

    // Layer 2: Dense (hiddenSize → hiddenSize)
    private readonly float[] _w2;       // [hiddenSize, hiddenSize]
    private readonly float[] _b2;       // [hiddenSize]

    // LayerNorm 2
    private readonly float[] _gamma2;   // [hiddenSize]
    private readonly float[] _beta2;    // [hiddenSize]

    // Layer 3: Dense (hiddenSize → hiddenSize)
    private readonly float[] _w3;       // [hiddenSize, hiddenSize]
    private readonly float[] _b3;       // [hiddenSize]

    // LayerNorm 3
    private readonly float[] _gamma3;   // [hiddenSize]
    private readonly float[] _beta3;    // [hiddenSize]

    // Layer 4: Dense (hiddenSize → 1)
    private readonly float[] _w4;       // [hiddenSize]
    private float _b4;

    // Cached activations for backprop
    private readonly float[] _z1;       // Dense1 output (pre-LN)
    private readonly float[] _xhat1;    // LN1 normalized
    private readonly float[] _ln1;      // LN1 output
    private readonly float[] _a1;       // post-ReLU1

    private readonly float[] _z2;       // Dense2 output (pre-LN)
    private readonly float[] _xhat2;    // LN2 normalized
    private readonly float[] _ln2;      // LN2 output
    private readonly float[] _a2;       // post-ReLU2

    private readonly float[] _z3;       // Dense3 output (pre-LN)
    private readonly float[] _xhat3;    // LN3 normalized
    private readonly float[] _ln3;      // LN3 output
    private readonly float[] _a3;       // post-ReLU3

    private float _invStd1, _invStd2, _invStd3;

    public WakeWordClassifier(int featureFrames, int embeddingDim, int hiddenSize = 128)
    {
        _inputSize = featureFrames * embeddingDim;
        _hiddenSize = hiddenSize;

        _w1 = new float[_hiddenSize * _inputSize];
        _b1 = new float[_hiddenSize];
        _gamma1 = new float[_hiddenSize];
        _beta1 = new float[_hiddenSize];

        _w2 = new float[_hiddenSize * _hiddenSize];
        _b2 = new float[_hiddenSize];
        _gamma2 = new float[_hiddenSize];
        _beta2 = new float[_hiddenSize];

        _w3 = new float[_hiddenSize * _hiddenSize];
        _b3 = new float[_hiddenSize];
        _gamma3 = new float[_hiddenSize];
        _beta3 = new float[_hiddenSize];

        _w4 = new float[_hiddenSize];
        _b4 = 0f;

        _z1 = new float[_hiddenSize];
        _xhat1 = new float[_hiddenSize];
        _ln1 = new float[_hiddenSize];
        _a1 = new float[_hiddenSize];

        _z2 = new float[_hiddenSize];
        _xhat2 = new float[_hiddenSize];
        _ln2 = new float[_hiddenSize];
        _a2 = new float[_hiddenSize];

        _z3 = new float[_hiddenSize];
        _xhat3 = new float[_hiddenSize];
        _ln3 = new float[_hiddenSize];
        _a3 = new float[_hiddenSize];

        InitializeWeights();
    }

    /// <summary>He initialization for dense weights, ones/zeros for LayerNorm.</summary>
    private void InitializeWeights()
    {
        var rng = new Random(42);

        var scale1 = MathF.Sqrt(2f / _inputSize);
        for (var i = 0; i < _w1.Length; i++)
            _w1[i] = (float)(rng.NextDouble() * 2 - 1) * scale1;

        var scale2 = MathF.Sqrt(2f / _hiddenSize);
        for (var i = 0; i < _w2.Length; i++)
            _w2[i] = (float)(rng.NextDouble() * 2 - 1) * scale2;

        for (var i = 0; i < _w3.Length; i++)
            _w3[i] = (float)(rng.NextDouble() * 2 - 1) * scale2;

        for (var i = 0; i < _w4.Length; i++)
            _w4[i] = (float)(rng.NextDouble() * 2 - 1) * scale2;

        // LayerNorm: gamma = 1, beta = 0
        Array.Fill(_gamma1, 1f);
        Array.Fill(_gamma2, 1f);
        Array.Fill(_gamma3, 1f);
        // beta1, beta2 are already zero-initialized
    }

    /// <summary>Forward pass: returns predicted probability.</summary>
    public float Forward(float[] input)
    {
        if (input.Length != _inputSize)
            throw new ArgumentException($"Expected input length {_inputSize}, got {input.Length}");

        // Layer 1: Dense
        for (var h = 0; h < _hiddenSize; h++)
        {
            var sum = _b1[h];
            var offset = h * _inputSize;
            for (var i = 0; i < _inputSize; i++)
                sum += _w1[offset + i] * input[i];
            _z1[h] = sum;
        }

        // LayerNorm 1
        LayerNormForward(_z1, _gamma1, _beta1, _xhat1, _ln1, out _invStd1);

        // ReLU 1
        for (var h = 0; h < _hiddenSize; h++)
            _a1[h] = _ln1[h] > 0 ? _ln1[h] : 0;

        // Layer 2: Dense
        for (var h = 0; h < _hiddenSize; h++)
        {
            var sum = _b2[h];
            var offset = h * _hiddenSize;
            for (var i = 0; i < _hiddenSize; i++)
                sum += _w2[offset + i] * _a1[i];
            _z2[h] = sum;
        }

        // LayerNorm 2
        LayerNormForward(_z2, _gamma2, _beta2, _xhat2, _ln2, out _invStd2);

        // ReLU 2
        for (var h = 0; h < _hiddenSize; h++)
            _a2[h] = _ln2[h] > 0 ? _ln2[h] : 0;

        // Layer 3: Dense
        for (var h = 0; h < _hiddenSize; h++)
        {
            var sum = _b3[h];
            var offset = h * _hiddenSize;
            for (var i = 0; i < _hiddenSize; i++)
                sum += _w3[offset + i] * _a2[i];
            _z3[h] = sum;
        }

        // LayerNorm 3
        LayerNormForward(_z3, _gamma3, _beta3, _xhat3, _ln3, out _invStd3);

        // ReLU 3
        for (var h = 0; h < _hiddenSize; h++)
            _a3[h] = _ln3[h] > 0 ? _ln3[h] : 0;

        // Layer 4: Dense → Sigmoid
        var z4 = _b4;
        for (var h = 0; h < _hiddenSize; h++)
            z4 += _w4[h] * _a3[h];

        return Sigmoid(z4);
    }

    private void LayerNormForward(float[] z, float[] gamma, float[] beta,
        float[] xhat, float[] output, out float invStd)
    {
        var mean = 0f;
        for (var i = 0; i < _hiddenSize; i++)
            mean += z[i];
        mean /= _hiddenSize;

        var variance = 0f;
        for (var i = 0; i < _hiddenSize; i++)
        {
            var diff = z[i] - mean;
            variance += diff * diff;
        }
        variance /= _hiddenSize;

        invStd = 1f / MathF.Sqrt(variance + LayerNormEps);
        for (var i = 0; i < _hiddenSize; i++)
        {
            xhat[i] = (z[i] - mean) * invStd;
            output[i] = gamma[i] * xhat[i] + beta[i];
        }
    }

    /// <summary>
    /// Train one step of binary cross-entropy loss with backpropagation.
    /// Returns the loss value.
    /// </summary>
    public float TrainStep(float[] input, float label, float learningRate, float positiveWeight, float negativeWeight, float weightDecay)
    {
        var prediction = Forward(input);

        var sampleWeight = label >= 0.5f ? positiveWeight : negativeWeight;

        // Binary cross-entropy loss
        var eps = 1e-7f;
        var loss = -sampleWeight * (label * MathF.Log(prediction + eps) + (1 - label) * MathF.Log(1 - prediction + eps));

        // dL/dz4 = prediction - label (sigmoid + BCE simplification)
        var dz4 = (prediction - label) * sampleWeight;

        // ── Backprop through Layer 4 ──
        var da3 = new float[_hiddenSize];
        for (var h = 0; h < _hiddenSize; h++)
            da3[h] = dz4 * _w4[h];

        for (var h = 0; h < _hiddenSize; h++)
            _w4[h] -= learningRate * ClipGradient(dz4 * _a3[h] + weightDecay * _w4[h]);
        _b4 -= learningRate * ClipGradient(dz4);

        // ── Backprop through ReLU 3 ──
        var dln3 = new float[_hiddenSize];
        for (var h = 0; h < _hiddenSize; h++)
            dln3[h] = _ln3[h] > 0 ? da3[h] : 0;

        // ── Backprop through LayerNorm 3 ──
        var dz3 = LayerNormBackward(dln3, _gamma3, _beta3, _xhat3, _invStd3, learningRate);

        // ── Backprop through Layer 3 ──
        var da2 = new float[_hiddenSize];
        for (var h = 0; h < _hiddenSize; h++)
            for (var j = 0; j < _hiddenSize; j++)
                da2[h] += dz3[j] * _w3[j * _hiddenSize + h];

        for (var h = 0; h < _hiddenSize; h++)
        {
            var offset = h * _hiddenSize;
            for (var i = 0; i < _hiddenSize; i++)
                _w3[offset + i] -= learningRate * ClipGradient(dz3[h] * _a2[i] + weightDecay * _w3[offset + i]);
            _b3[h] -= learningRate * ClipGradient(dz3[h]);
        }

        // ── Backprop through ReLU 2 ──
        var dln2 = new float[_hiddenSize];
        for (var h = 0; h < _hiddenSize; h++)
            dln2[h] = _ln2[h] > 0 ? da2[h] : 0;

        // ── Backprop through LayerNorm 2 ──
        var dz2 = LayerNormBackward(dln2, _gamma2, _beta2, _xhat2, _invStd2, learningRate);

        // ── Backprop through Layer 2 ──
        var da1 = new float[_hiddenSize];
        for (var h = 0; h < _hiddenSize; h++)
            for (var j = 0; j < _hiddenSize; j++)
                da1[h] += dz2[j] * _w2[j * _hiddenSize + h];

        for (var h = 0; h < _hiddenSize; h++)
        {
            var offset = h * _hiddenSize;
            for (var i = 0; i < _hiddenSize; i++)
                _w2[offset + i] -= learningRate * ClipGradient(dz2[h] * _a1[i] + weightDecay * _w2[offset + i]);
            _b2[h] -= learningRate * ClipGradient(dz2[h]);
        }

        // ── Backprop through ReLU 1 ──
        var dln1 = new float[_hiddenSize];
        for (var h = 0; h < _hiddenSize; h++)
            dln1[h] = _ln1[h] > 0 ? da1[h] : 0;

        // ── Backprop through LayerNorm 1 ──
        var dz1 = LayerNormBackward(dln1, _gamma1, _beta1, _xhat1, _invStd1, learningRate);

        // ── Backprop through Layer 1 ──
        for (var h = 0; h < _hiddenSize; h++)
        {
            var offset = h * _inputSize;
            for (var i = 0; i < _inputSize; i++)
                _w1[offset + i] -= learningRate * ClipGradient(dz1[h] * input[i] + weightDecay * _w1[offset + i]);
            _b1[h] -= learningRate * ClipGradient(dz1[h]);
        }

        return loss;
    }

    private float[] LayerNormBackward(float[] dout, float[] gamma, float[] beta,
        float[] xhat, float invStd, float learningRate)
    {
        var N = (float)_hiddenSize;

        // Compute dxhat using current gamma BEFORE updating
        var dxhat = new float[_hiddenSize];
        for (var i = 0; i < _hiddenSize; i++)
            dxhat[i] = dout[i] * gamma[i];

        // Update gamma and beta
        for (var i = 0; i < _hiddenSize; i++)
        {
            gamma[i] -= learningRate * ClipGradient(dout[i] * xhat[i]);
            beta[i] -= learningRate * ClipGradient(dout[i]);
        }

        // Gradient w.r.t. input z
        var sumDxhat = 0f;
        var sumDxhatXhat = 0f;
        for (var i = 0; i < _hiddenSize; i++)
        {
            sumDxhat += dxhat[i];
            sumDxhatXhat += dxhat[i] * xhat[i];
        }

        var dz = new float[_hiddenSize];
        for (var i = 0; i < _hiddenSize; i++)
            dz[i] = invStd * (dxhat[i] - sumDxhat / N - xhat[i] * sumDxhatXhat / N);

        return dz;
    }

    private const float GradientClipValue = 5.0f;

    private static float ClipGradient(float gradient)
    {
        if (float.IsNaN(gradient) || float.IsInfinity(gradient))
            return 0f;
        return Math.Clamp(gradient, -GradientClipValue, GradientClipValue);
    }

    /// <summary>
    /// Get all model weights as a serializable structure for ONNX export.
    /// </summary>
    public ModelWeights GetWeights()
    {
        return new ModelWeights
        {
            InputSize = _inputSize,
            HiddenSize = _hiddenSize,
            W1 = (float[])_w1.Clone(),
            B1 = (float[])_b1.Clone(),
            Gamma1 = (float[])_gamma1.Clone(),
            Beta1 = (float[])_beta1.Clone(),
            W2 = (float[])_w2.Clone(),
            B2 = (float[])_b2.Clone(),
            Gamma2 = (float[])_gamma2.Clone(),
            Beta2 = (float[])_beta2.Clone(),
            W3 = (float[])_w3.Clone(),
            B3 = (float[])_b3.Clone(),
            Gamma3 = (float[])_gamma3.Clone(),
            Beta3 = (float[])_beta3.Clone(),
            W4 = (float[])_w4.Clone(),
            B4 = _b4
        };
    }

    private static float Sigmoid(float x)
    {
        if (x >= 0)
            return 1f / (1f + MathF.Exp(-x));
        var ex = MathF.Exp(x);
        return ex / (1f + ex);
    }
}

/// <summary>
/// Serializable model weights for ONNX export.
/// </summary>
public sealed class ModelWeights
{
    public int InputSize { get; set; }
    public int HiddenSize { get; set; }
    public float[] W1 { get; set; } = [];
    public float[] B1 { get; set; } = [];
    public float[] Gamma1 { get; set; } = [];
    public float[] Beta1 { get; set; } = [];
    public float[] W2 { get; set; } = [];
    public float[] B2 { get; set; } = [];
    public float[] Gamma2 { get; set; } = [];
    public float[] Beta2 { get; set; } = [];
    public float[] W3 { get; set; } = [];
    public float[] B3 { get; set; } = [];
    public float[] Gamma3 { get; set; } = [];
    public float[] Beta3 { get; set; } = [];
    public float[] W4 { get; set; } = [];
    public float B4 { get; set; }
}
