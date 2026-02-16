using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace AwakeWord.Core;

/// <summary>
/// Provides functionality to load audio files (WAV, MP3, etc.) and convert them
/// to the required format for wake-word detection (16-bit mono PCM at 16 kHz).
/// </summary>
public static class AudioFileLoader
{
    /// <summary>
    /// Loads an audio file (WAV, MP3, etc.) and converts it to 16-bit mono PCM at 16 kHz.
    /// </summary>
    /// <param name="filePath">Path to the audio file to load.</param>
    /// <returns>Array of 16-bit PCM samples at 16 kHz mono.</returns>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the file format is not supported.</exception>
    public static short[] LoadAudioFile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Audio file not found: {filePath}");
        }

        // Try reading as 16-bit PCM WAV first for optimal performance
        if (TryLoadDirectPcm(filePath, out var directSamples))
        {
            return directSamples;
        }

        // Use AudioFileReader for format conversion (supports WAV, MP3, AIFF, etc.)
        return LoadWithConversion(filePath);
    }

    /// <summary>
    /// Attempts to load a WAV file directly if it's already in the correct format
    /// (16-bit PCM mono at 16 kHz).
    /// </summary>
    private static bool TryLoadDirectPcm(string filePath, out short[] samples)
    {
        samples = Array.Empty<short>();

        // Only try direct loading for WAV files
        if (!filePath.EndsWith(".wav", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        try
        {
            using var reader = new WaveFileReader(filePath);
            if (reader.WaveFormat.Encoding == WaveFormatEncoding.Pcm &&
                reader.WaveFormat.BitsPerSample == 16 &&
                reader.WaveFormat.SampleRate == 16000 &&
                reader.WaveFormat.Channels == 1)
            {
                // Perfect match - read directly
                var sampleList = new List<short>();
                var buffer = new byte[reader.WaveFormat.AverageBytesPerSecond];
                int bytesRead;

                while ((bytesRead = reader.Read(buffer, 0, buffer.Length)) > 0)
                {
                    for (var i = 0; i < bytesRead; i += 2)
                    {
                        if (i + 1 < bytesRead)
                        {
                            var sample = (short)(buffer[i] | (buffer[i + 1] << 8));
                            sampleList.Add(sample);
                        }
                    }
                }

                samples = sampleList.ToArray();
                return true;
            }
        }
        catch
        {
            // Fall through to conversion method
        }

        return false;
    }

    /// <summary>
    /// Loads an audio file using NAudio's AudioFileReader, then resamples to
    /// 16 kHz mono using WDL resampler + stereo-to-mono conversion as needed.
    /// </summary>
    private static short[] LoadWithConversion(string filePath)
    {
        try
        {
            using var audioReader = new AudioFileReader(filePath);

            // Build a pipeline: decode → mono → resample to 16 kHz
            ISampleProvider pipeline = audioReader;

            // Convert to mono if stereo (or more channels)
            if (audioReader.WaveFormat.Channels == 2)
            {
                pipeline = new StereoToMonoSampleProvider(pipeline)
                {
                    LeftVolume = 0.5f,
                    RightVolume = 0.5f
                };
            }
            else if (audioReader.WaveFormat.Channels > 2)
            {
                // For multi-channel audio, use MultiplexingSampleProvider to take first channel
                var mono = new MultiplexingSampleProvider(new[] { pipeline }, 1);
                mono.ConnectInputToOutput(0, 0);
                pipeline = mono;
            }

            // Resample to 16 kHz if not already
            if (pipeline.WaveFormat.SampleRate != AudioFormat.SampleRateHz)
            {
                pipeline = new WdlResamplingSampleProvider(pipeline, AudioFormat.SampleRateHz);
            }

            var samples = new List<short>();
            var floatBuffer = new float[AudioFormat.SampleRateHz]; // 1-second read buffer
            int samplesRead;

            while ((samplesRead = pipeline.Read(floatBuffer, 0, floatBuffer.Length)) > 0)
            {
                for (var i = 0; i < samplesRead; i++)
                {
                    // Clamp and convert float [-1, 1] to short [-32768, 32767]
                    var sample = (short)(Math.Clamp(floatBuffer[i], -1f, 1f) * 32767f);
                    samples.Add(sample);
                }
            }

            return samples.ToArray();
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to load audio file '{filePath}'. The file format may not be supported.", ex);
        }
    }

    /// <summary>
    /// Checks if a file extension is supported for audio loading.
    /// </summary>
    /// <param name="fileExtension">The file extension (e.g., ".wav", ".mp3").</param>
    /// <returns>True if the extension is supported, false otherwise.</returns>
    public static bool IsSupportedFormat(string fileExtension)
    {
        var normalizedExtension = fileExtension.ToLowerInvariant();
        if (!normalizedExtension.StartsWith('.'))
        {
            normalizedExtension = "." + normalizedExtension;
        }

        // Supported formats via NAudio's AudioFileReader (using Windows Media Foundation)
        return normalizedExtension is ".wav" or ".mp3" or ".aiff" or ".aif" or ".m4a" or ".wma";
    }
}
