using NAudio.Wave;

namespace AwakeWord.Training;

/// <summary>
/// Records audio from the microphone and returns 16-bit PCM mono @ 16 kHz samples.
/// </summary>
public sealed class AudioRecorder : IDisposable
{
    private WaveInEvent? _waveIn;
    private MemoryStream? _recordingStream;
    private readonly object _lock = new();

    /// <summary>Whether the recorder is currently recording.</summary>
    public bool IsRecording { get; private set; }

    /// <summary>Raised when audio data is available during recording (for level meters).</summary>
    public event Action<float>? OnLevelChanged;

    /// <summary>Start recording from the default microphone.</summary>
    public void StartRecording()
    {
        if (IsRecording) return;

        _recordingStream = new MemoryStream();
        _waveIn = new WaveInEvent
        {
            WaveFormat = new WaveFormat(16000, 16, 1), // 16 kHz, 16-bit, mono
            BufferMilliseconds = 80
        };

        _waveIn.DataAvailable += (_, e) =>
        {
            lock (_lock)
            {
                _recordingStream?.Write(e.Buffer, 0, e.BytesRecorded);
            }

            // Calculate RMS level for UI feedback
            var level = CalculateRmsLevel(e.Buffer, e.BytesRecorded);
            OnLevelChanged?.Invoke(level);
        };

        _waveIn.StartRecording();
        IsRecording = true;
    }

    /// <summary>Stop recording and return the captured PCM samples.</summary>
    public short[] StopRecording()
    {
        if (!IsRecording || _waveIn == null) return Array.Empty<short>();

        _waveIn.StopRecording();
        _waveIn.Dispose();
        _waveIn = null;
        IsRecording = false;

        byte[] rawBytes;
        lock (_lock)
        {
            rawBytes = _recordingStream?.ToArray() ?? Array.Empty<byte>();
            _recordingStream?.Dispose();
            _recordingStream = null;
        }

        // Convert bytes to shorts
        var samples = new short[rawBytes.Length / 2];
        Buffer.BlockCopy(rawBytes, 0, samples, 0, rawBytes.Length);
        return samples;
    }

    /// <summary>Get available recording devices.</summary>
    public static List<string> GetRecordingDevices()
    {
        var devices = new List<string>();
        for (var i = 0; i < WaveInEvent.DeviceCount; i++)
        {
            var caps = WaveInEvent.GetCapabilities(i);
            devices.Add(caps.ProductName);
        }
        return devices;
    }

    private static float CalculateRmsLevel(byte[] buffer, int bytesRecorded)
    {
        var sumOfSquares = 0.0;
        var sampleCount = bytesRecorded / 2;
        for (var i = 0; i < bytesRecorded; i += 2)
        {
            if (i + 1 >= bytesRecorded) break;
            var sample = (short)(buffer[i] | (buffer[i + 1] << 8));
            sumOfSquares += sample * sample;
        }
        var rms = Math.Sqrt(sumOfSquares / sampleCount);
        return (float)(rms / 32768.0); // Normalize to 0..1
    }

    public void Dispose()
    {
        if (IsRecording)
        {
            _waveIn?.StopRecording();
            _waveIn?.Dispose();
        }
        _recordingStream?.Dispose();
    }
}
