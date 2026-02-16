Goals
- Build a cross-platform C# wake-word detection library with a test application.
- Provide configurable wake-word detection; support multiple wake words (jarvis, mycroft).
- Deliver fast, low-latency detection on low-power devices.
- Support loading audio from files in multiple formats (WAV, MP3, M4A, AIFF, WMA).

Limitations
- Input audio must be mono, 16 kHz.
- Wake-word detection must use an ONNX model.
- Audio file loading uses NAudio with Windows Media Foundation (Windows only).

Other requirements
- All functions must be covered by unit tests using the Microsoft test framework.
- Each class must be in its own file.
- Organize classes into folders by domain or purpose.
- Provide a test application that captures audio from the microphone and reacts to the wake word.
- Provide a file processor application for batch processing of audio files.