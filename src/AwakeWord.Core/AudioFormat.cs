namespace AwakeWord.Core;

public static class AudioFormat
{
    public const int SampleRateHz = 16000;
    public const int Channels = 1;

    public static void Ensure(int sampleRateHz, int channels)
    {
        if (sampleRateHz != SampleRateHz)
        {
            throw new ArgumentException($"Expected {SampleRateHz} Hz audio.", nameof(sampleRateHz));
        }

        if (channels != Channels)
        {
            throw new ArgumentException($"Expected {Channels} channel audio.", nameof(channels));
        }
    }
}
