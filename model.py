import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Load pretrained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

model.eval()  # Set model to evaluation mode


def extract_embedding(audio_path):
    # Load audio file (supports .wav, .m4a, etc.)
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000
        )
        waveform = resampler(waveform)

    # Remove channel dimension
    waveform = waveform.squeeze().numpy()

    # Tokenize
    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding


if __name__ == "__main__":
    test_file = "test.wav"  # <-- Make sure this file exists in accent-ai folder

    embedding = extract_embedding(test_file)

    print("Embedding shape:", embedding.shape)
    print("Embedding sample:", embedding[0][:10])