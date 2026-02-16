namespace AwakeWord.Core;

/// <summary>
/// Detects a wake word in streaming audio chunks.
/// Implementations maintain internal state between calls.
/// Feed 16-bit PCM audio samples (mono, 16 kHz) progressively.
/// </summary>
public interface IAwakeWordDetector : IDisposable
{
    string WakeWord { get; }

    /// <summary>
    /// Process a chunk of 16-bit PCM audio samples and check for wake-word detection.
    /// </summary>
    /// <param name="audioSamples">Chunk of 16-bit PCM samples (mono, 16 kHz).</param>
    /// <param name="confidence">Detection confidence if the wake word was detected.</param>
    /// <returns>True if the wake word was detected in the latest window.</returns>
    bool ProcessAudio(ReadOnlySpan<short> audioSamples, out float confidence);

    /// <summary>
    /// Reset internal buffers (e.g., after a detection or silence period).
    /// </summary>
    void Reset();
}
