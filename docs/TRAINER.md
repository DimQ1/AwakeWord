# AwakeWord Trainer

GUI app for creating custom wake-word models from your own audio samples.

## Requirements

- .NET 10.0 or later
- Windows (WPF UI)
- ONNX models in the models directory:
  - melspectrogram.onnx
  - embedding_model.onnx

## Run

From the repo root:

```bash
dotnet run --project apps/AwakeWord.Trainer
```

## Workflow

1. Set the Models Directory to the folder that contains the ONNX base models.
2. Choose the Output Model Path for the new wake word model.
3. Add or record Positive samples (contain the wake word).
4. Add or record Negative samples (background or other speech).
5. Click Start Training and wait for completion.
6. Test with Microphone or Test with File.

## Tips for Better Quality

- Use a similar number of positive and negative samples.
- Include varied background noise in negative samples.
- Record positives from different distances and speaking styles.
- Keep samples clear (no clipping) and at least 1.5s long.

## Output

The trainer exports a single ONNX model at the Output Model Path. You can use
it with AwakeWord.TestApp or AwakeWord.FileProcessor by pointing
WakeWordModelPath to the generated file.
