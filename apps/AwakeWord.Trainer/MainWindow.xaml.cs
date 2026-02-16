using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Threading;
using AwakeWord.Core;
using AwakeWord.Training;
using Microsoft.Win32;
using NAudio.Wave;

namespace AwakeWord.Trainer;

public partial class MainWindow : Window
{
    private readonly ObservableCollection<string> _positiveFiles = new();
    private readonly ObservableCollection<string> _negativeFiles = new();

    private AudioRecorder? _recorder;
    private CancellationTokenSource? _trainingCts;
    private string? _trainedModelPath;

    private WaveInEvent? _testWaveIn;
    private OnnxWakeWordDetector? _testDetector;
    private bool _isTesting;
    private bool _testDetected;
    private float _testMaxConfidence;

    private int _positiveRecordingCount;
    private int _negativeRecordingCount;

    public MainWindow()
    {
        InitializeComponent();
        PositiveFilesList.ItemsSource = _positiveFiles;
        NegativeFilesList.ItemsSource = _negativeFiles;
    }

    // ── Configuration ──

    private void BrowseModelsDir_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog
        {
            Title = "Select Models Directory"
        };

        if (dialog.ShowDialog() == true)
            ModelsDirBox.Text = dialog.FolderName;
    }

    private void BrowseOutputPath_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new SaveFileDialog
        {
            Title = "Save Trained Model",
            Filter = "ONNX Model|*.onnx",
            DefaultExt = ".onnx",
            FileName = $"{WakeWordNameBox.Text}.onnx"
        };

        if (dialog.ShowDialog() == true)
            OutputPathBox.Text = dialog.FileName;
    }

    // ── Positive Samples ──

    private void AddPositiveFiles_Click(object sender, RoutedEventArgs e)
    {
        var files = BrowseAudioFiles("Select Positive Audio Files (containing wake word)");
        foreach (var f in files)
            _positiveFiles.Add(f);
    }

    private void RemovePositiveFiles_Click(object sender, RoutedEventArgs e)
    {
        var selected = PositiveFilesList.SelectedItems.Cast<string>().ToList();
        foreach (var s in selected)
            _positiveFiles.Remove(s);
    }

    private void RecordPositive_Click(object sender, RoutedEventArgs e)
    {
        if (_recorder is { IsRecording: true })
        {
            // Stop recording
            var samples = _recorder.StopRecording();
            _recorder.OnLevelChanged -= OnPositiveLevelChanged;
            _recorder.Dispose();
            _recorder = null;

            if (samples.Length > 0)
            {
                _positiveRecordingCount++;
                var filePath = SaveRecordingToWav(samples, "positive", _positiveRecordingCount);
                _positiveFiles.Add(filePath);
                AppendLog($"Saved positive recording: {Path.GetFileName(filePath)}");
            }

            RecordPositiveBtn.Content = "🎙 Record";
            PositiveRecordingStatus.Text = "";
            PositiveLevelBar.Value = 0;
            RecordNegativeBtn.IsEnabled = true;
            return;
        }

        _recorder = new AudioRecorder();
        _recorder.OnLevelChanged += OnPositiveLevelChanged;
        _recorder.StartRecording();

        RecordPositiveBtn.Content = "⏹ Stop";
        PositiveRecordingStatus.Text = "Recording...";
        RecordNegativeBtn.IsEnabled = false;
    }

    private void OnPositiveLevelChanged(float level)
    {
        Dispatcher.BeginInvoke(() => PositiveLevelBar.Value = level);
    }

    // ── Negative Samples ──

    private void AddNegativeFiles_Click(object sender, RoutedEventArgs e)
    {
        var files = BrowseAudioFiles("Select Negative Audio Files (no wake word)");
        foreach (var f in files)
            _negativeFiles.Add(f);
    }

    private void RemoveNegativeFiles_Click(object sender, RoutedEventArgs e)
    {
        var selected = NegativeFilesList.SelectedItems.Cast<string>().ToList();
        foreach (var s in selected)
            _negativeFiles.Remove(s);
    }

    private void RecordNegative_Click(object sender, RoutedEventArgs e)
    {
        if (_recorder is { IsRecording: true })
        {
            var samples = _recorder.StopRecording();
            _recorder.OnLevelChanged -= OnNegativeLevelChanged;
            _recorder.Dispose();
            _recorder = null;

            if (samples.Length > 0)
            {
                _negativeRecordingCount++;
                var filePath = SaveRecordingToWav(samples, "negative", _negativeRecordingCount);
                _negativeFiles.Add(filePath);
                AppendLog($"Saved negative recording: {Path.GetFileName(filePath)}");
            }

            RecordNegativeBtn.Content = "🎙 Record";
            NegativeRecordingStatus.Text = "";
            NegativeLevelBar.Value = 0;
            RecordPositiveBtn.IsEnabled = true;
            return;
        }

        _recorder = new AudioRecorder();
        _recorder.OnLevelChanged += OnNegativeLevelChanged;
        _recorder.StartRecording();

        RecordNegativeBtn.Content = "⏹ Stop";
        NegativeRecordingStatus.Text = "Recording...";
        RecordPositiveBtn.IsEnabled = false;
    }

    private void OnNegativeLevelChanged(float level)
    {
        Dispatcher.BeginInvoke(() => NegativeLevelBar.Value = level);
    }

    // ── Training ──

    private async void Train_Click(object sender, RoutedEventArgs e)
    {
        // Validate
        var modelsDir = ModelsDirBox.Text.Trim();
        var melPath = Path.Combine(modelsDir, "melspectrogram.onnx");
        var embPath = Path.Combine(modelsDir, "embedding_model.onnx");

        if (!File.Exists(melPath) || !File.Exists(embPath))
        {
            MessageBox.Show(
                $"Models not found in '{modelsDir}'.\nExpected melspectrogram.onnx and embedding_model.onnx.",
                "Missing Models", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        var positiveAudioFiles = _positiveFiles.ToList();
        var negativeAudioFiles = _negativeFiles.ToList();

        var totalPositive = positiveAudioFiles.Count;
        var totalNegative = negativeAudioFiles.Count;

        if (totalPositive == 0)
        {
            MessageBox.Show("Add at least one positive audio sample (containing the wake word).",
                "No Positive Samples", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (totalNegative == 0)
        {
            MessageBox.Show("Add at least one negative audio sample (without the wake word).",
                "No Negative Samples", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (!int.TryParse(EpochsBox.Text, out var epochs) || epochs <= 0)
        {
            MessageBox.Show("Epochs must be a positive integer.", "Invalid Input",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (!float.TryParse(LearningRateBox.Text, out var lr) || lr <= 0)
        {
            MessageBox.Show("Learning rate must be a positive number.", "Invalid Input",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        // Set up training
        var config = new TrainingConfig
        {
            MelSpectrogramModelPath = melPath,
            EmbeddingModelPath = embPath,
            WakeWordName = WakeWordNameBox.Text.Trim(),
            OutputModelPath = OutputPathBox.Text.Trim(),
            Epochs = epochs,
            LearningRate = lr
        };

        SetTrainingUiState(isTraining: true);
        LogBox.Clear();
        _trainingCts = new CancellationTokenSource();

        try
        {
            using var trainer = new WakeWordTrainer(config);
            trainer.OnLog += msg => Dispatcher.BeginInvoke(() => AppendLog(msg));
            trainer.OnProgress += progress => Dispatcher.BeginInvoke(() =>
            {
                TrainingProgressBar.Value = (double)progress.Epoch / progress.TotalEpochs * 100;
                TrainingStatusText.Text = $"Epoch {progress.Epoch}/{progress.TotalEpochs} — " +
                                          $"Loss: {progress.Loss:F4}, Accuracy: {progress.Accuracy:P1}";
            });

            _trainedModelPath = await trainer.TrainAsync(
                positiveAudioFiles, negativeAudioFiles, _trainingCts.Token);

            AppendLog($"\n✅ Model saved to: {_trainedModelPath}");
            TrainingStatusText.Text = "Training complete!";
            TestButton.IsEnabled = true;
            TestFileButton.IsEnabled = true;

            MessageBox.Show($"Model trained and saved to:\n{_trainedModelPath}",
                "Training Complete", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        catch (OperationCanceledException)
        {
            AppendLog("\n⚠ Training cancelled.");
            TrainingStatusText.Text = "Cancelled.";
        }
        catch (Exception ex)
        {
            AppendLog($"\n❌ Error: {ex.Message}");
            TrainingStatusText.Text = "Error!";
            MessageBox.Show(ex.Message, "Training Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            SetTrainingUiState(isTraining: false);
        }
    }

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        _trainingCts?.Cancel();
    }

    // ── Testing ──

    private void Test_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(_trainedModelPath) || !File.Exists(_trainedModelPath))
        {
            MessageBox.Show("Train a model first.", "No Model", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (_isTesting)
            return;

        TestButton.IsEnabled = false;
        TestFileButton.IsEnabled = false;
        EndTestButton.IsEnabled = true;
        TestResultText.Text = "Listening... Say your wake word!";
        AppendLog("\n🎤 Testing: Listening until you stop...");

        try
        {
            var modelsDir = ModelsDirBox.Text.Trim();
            var config = new WakeWordConfig
            {
                MelSpectrogramModelPath = Path.Combine(modelsDir, "melspectrogram.onnx"),
                EmbeddingModelPath = Path.Combine(modelsDir, "embedding_model.onnx"),
                WakeWordModelPath = _trainedModelPath,
                WakeWord = WakeWordNameBox.Text.Trim(),
                DetectionThreshold = 0.5f
            };

            _testDetector = new OnnxWakeWordDetector(config);
            _testWaveIn = new WaveInEvent
            {
                WaveFormat = new WaveFormat(16000, 16, 1),
                BufferMilliseconds = 80
            };

            _testDetected = false;
            _testMaxConfidence = 0f;
            _isTesting = true;

            _testWaveIn.DataAvailable += TestWaveInOnDataAvailable;
            _testWaveIn.StartRecording();
        }
        catch (Exception ex)
        {
            TestResultText.Text = $"Error: {ex.Message}";
            AppendLog($"  ❌ Test error: {ex.Message}");
            StopMicrophoneTest(updateUi: true);
        }
    }

    private void EndTest_Click(object sender, RoutedEventArgs e)
    {
        StopMicrophoneTest(updateUi: true);
    }

    private void TestWaveInOnDataAvailable(object? sender, WaveInEventArgs args)
    {
        if (!_isTesting || _testDetector == null)
            return;

        var samples = new short[args.BytesRecorded / 2];
        Buffer.BlockCopy(args.Buffer, 0, samples, 0, args.BytesRecorded);

        if (_testDetector.ProcessAudio(samples, out var confidence))
        {
            _testDetected = true;
            _testMaxConfidence = Math.Max(_testMaxConfidence, confidence);
            Dispatcher.BeginInvoke(() =>
            {
                TestResultText.Text = $"✅ Detected! Confidence: {confidence:P1}";
                AppendLog($"  ✅ Wake word detected with confidence {confidence:F4}");
            });
        }
        else if (confidence > _testMaxConfidence)
        {
            _testMaxConfidence = confidence;
        }
    }

    private void StopMicrophoneTest(bool updateUi)
    {
        if (!_isTesting)
            return;

        _isTesting = false;

        if (_testWaveIn != null)
        {
            _testWaveIn.DataAvailable -= TestWaveInOnDataAvailable;
            _testWaveIn.StopRecording();
            _testWaveIn.Dispose();
            _testWaveIn = null;
        }

        _testDetector?.Dispose();
        _testDetector = null;

        if (!_testDetected)
        {
            TestResultText.Text = $"No detection. Max confidence: {_testMaxConfidence:P1}";
            AppendLog($"  No detection. Max confidence was {_testMaxConfidence:F4}");
        }

        if (updateUi)
        {
            TestButton.IsEnabled = true;
            TestFileButton.IsEnabled = true;
            EndTestButton.IsEnabled = false;
        }
    }

    protected override void OnClosed(EventArgs e)
    {
        StopMicrophoneTest(updateUi: false);
        base.OnClosed(e);
    }

    private void TestFile_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(_trainedModelPath) || !File.Exists(_trainedModelPath))
        {
            MessageBox.Show("Train a model first.", "No Model", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        var dialog = new OpenFileDialog
        {
            Title = "Select Audio File to Test",
            Filter = "Audio Files|*.wav;*.mp3;*.m4a;*.aiff;*.wma|All Files|*.*"
        };

        if (dialog.ShowDialog() != true) return;

        try
        {
            var modelsDir = ModelsDirBox.Text.Trim();
            var config = new WakeWordConfig
            {
                MelSpectrogramModelPath = Path.Combine(modelsDir, "melspectrogram.onnx"),
                EmbeddingModelPath = Path.Combine(modelsDir, "embedding_model.onnx"),
                WakeWordModelPath = _trainedModelPath,
                WakeWord = WakeWordNameBox.Text.Trim(),
                DetectionThreshold = 0.5f
            };

            using var detector = new OnnxWakeWordDetector(config);
            var samples = AudioFileLoader.LoadAudioFile(dialog.FileName);
            AppendLog($"\n📁 Testing file: {Path.GetFileName(dialog.FileName)} ({samples.Length / 16000.0:F1}s)");

            var detections = 0;
            var maxConfidence = 0f;
            const int chunkSize = 1280;

            for (var offset = 0; offset < samples.Length; offset += chunkSize)
            {
                var length = Math.Min(chunkSize, samples.Length - offset);
                var chunk = new short[length];
                Array.Copy(samples, offset, chunk, 0, length);

                if (detector.ProcessAudio(chunk, out var confidence))
                {
                    detections++;
                    maxConfidence = Math.Max(maxConfidence, confidence);
                    AppendLog($"  Detection at {offset / 16000.0:F2}s with confidence {confidence:F4}");
                }
                else if (confidence > maxConfidence)
                {
                    maxConfidence = confidence;
                }
            }

            TestResultText.Text = detections > 0
                ? $"✅ {detections} detection(s), max confidence: {maxConfidence:P1}"
                : $"No detection. Max confidence: {maxConfidence:P1}";

            AppendLog($"  Result: {detections} detection(s), max confidence: {maxConfidence:F4}");
        }
        catch (Exception ex)
        {
            TestResultText.Text = $"Error: {ex.Message}";
            AppendLog($"  ❌ Error: {ex.Message}");
        }
    }

    // ── Helpers ──

    private static string[] BrowseAudioFiles(string title)
    {
        var dialog = new OpenFileDialog
        {
            Title = title,
            Filter = "Audio Files|*.wav;*.mp3;*.m4a;*.aiff;*.wma|All Files|*.*",
            Multiselect = true
        };

        return dialog.ShowDialog() == true ? dialog.FileNames : Array.Empty<string>();
    }

    private void SetTrainingUiState(bool isTraining)
    {
        TrainButton.IsEnabled = !isTraining;
        CancelButton.IsEnabled = isTraining;
        RecordPositiveBtn.IsEnabled = !isTraining;
        RecordNegativeBtn.IsEnabled = !isTraining;
    }

    private void AppendLog(string message)
    {
        LogBox.AppendText(message + Environment.NewLine);
        LogBox.ScrollToEnd();
    }

    private string SaveRecordingToWav(short[] samples, string subfolder, int index)
    {
        var outputDir = Path.GetDirectoryName(OutputPathBox.Text.Trim());
        if (string.IsNullOrEmpty(outputDir))
            outputDir = Directory.GetCurrentDirectory();

        var folder = Path.Combine(outputDir, subfolder);
        Directory.CreateDirectory(folder);

        var wakeWord = WakeWordNameBox.Text.Trim();
        var fileName = $"{wakeWord}_{subfolder}_{index:D3}.wav";
        var filePath = Path.Combine(folder, fileName);

        using var writer = new WaveFileWriter(filePath, new WaveFormat(16000, 16, 1));
        var bytes = new byte[samples.Length * 2];
        Buffer.BlockCopy(samples, 0, bytes, 0, bytes.Length);
        writer.Write(bytes, 0, bytes.Length);

        return filePath;
    }
}
