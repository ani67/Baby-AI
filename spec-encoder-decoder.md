# SPEC: Encoder / Decoder
*Component 3 of 9 — Translates between the world and the Baby Model*

---

## What it is

The boundary layer between raw experience and the Baby Model's internal world.

The Encoder takes something from the outside — an image, a video clip, a text
string — and produces a fixed-size vector the Baby Model can consume.

The Decoder takes a vector the Baby Model produces and turns it back into
human-readable text.

Neither component learns during the current phase. They use pretrained,
frozen weights. The Baby Model learns; the Encoder/Decoder just translate.

---

## Why this separation matters

The Baby Model should not have to learn "what is a pixel" or "what is a character."
Those are solved problems. CLIP already knows how to see. A tokenizer already
knows how to read. The Baby Model's job is to learn *relationships between concepts*
— and it can only do that if concepts arrive already in a common language.

That common language is a 512-dimensional embedding vector.
Everything — images, text, video frames — arrives as a 512-dim float tensor.
The Baby Model sees no difference between modalities.
It just sees vectors.

This is also why the Baby Model will naturally develop cross-modal clusters:
the "dog" image vector and the "dog" text vector will be close in CLIP space,
so they'll activate similar clusters, so those clusters will wire together.
Grounded language emerges from the geometry, not from explicit labeling.

---

## Location in the project

```
project/
  backend/
    encoder/
      encoder.py       ← ImageEncoder, TextEncoder, VideoEncoder
      decoder.py       ← TextDecoder
      clip_mlx.py      ← MLX wrapper for CLIP (M1 optimized)
      vocab.py         ← simple token vocabulary for the generative head
```

---

## The shared embedding space

All encoders output to the same space: **R^512, L2-normalized**.

L2 normalization is critical — it means cosine similarity equals dot product,
and all vectors live on the unit hypersphere. This makes:
- Distances meaningful across modalities
- The Baby Model's clustering work correctly
- CLIP's own geometry preserved (CLIP was trained this way)

```python
import torch
import torch.nn.functional as F

def normalize(v: torch.Tensor) -> torch.Tensor:
    return F.normalize(v, dim=-1)
```

Every encoder ends with this call. No exceptions.

---

## Components

---

### `ImageEncoder`

Wraps CLIP ViT-B/32 via MLX (Apple Silicon optimized).
Takes a PIL Image or raw bytes, returns a 512-dim normalized vector.

```python
class ImageEncoder:
    def __init__(self):
        """
        Loads CLIP ViT-B/32 weights.
        On first call, downloads from HuggingFace (~330MB).
        Cached to ~/.cache/huggingface after first download.
        Uses MLX backend for M1 GPU acceleration.
        """

    def encode(self, image: PIL.Image.Image) -> torch.Tensor:
        """
        Returns: tensor of shape (512,), L2-normalized, on CPU.
        
        Preprocessing applied internally:
          - Resize to 224x224 (CLIP input size)
          - Center crop
          - Normalize with CLIP's mean/std
          - Convert to RGB if not already
        """

    def encode_batch(self, images: list[PIL.Image.Image]) -> torch.Tensor:
        """
        Returns: tensor of shape (N, 512), each row L2-normalized.
        Faster than calling encode() N times — processes as a batch.
        """

    def encode_bytes(self, image_bytes: bytes) -> torch.Tensor:
        """
        Convenience method — decodes bytes to PIL then encodes.
        Used when images arrive as file uploads or base64 strings.
        """
```

**MLX vs PyTorch for CLIP:**
MLX's Apple Silicon optimization gives ~2-3x speedup over PyTorch MPS for CLIP
inference. The MLX CLIP implementation is in `clip_mlx.py`. Internally it uses
`mlx.core` tensors, but all public methods convert to CPU PyTorch tensors before
returning so the rest of the system stays in one tensor framework.

```python
# clip_mlx.py — internal shape
import mlx.core as mx
from transformers import CLIPModel, CLIPProcessor

class CLIPWrapper:
    """
    Loads CLIP via HuggingFace transformers.
    Runs forward pass via MLX.
    Returns numpy array, converted to torch by callers.
    """
```

---

### `TextEncoder`

Wraps CLIP's text encoder specifically.
Takes a string, returns a 512-dim normalized vector.

CLIP was trained on (image, text) pairs — the text and image encoders
were trained jointly to produce aligned embeddings. This means:
- "a photo of a dog" and [image of dog] are close in embedding space
- "a photo of a cat" and [image of cat] are close
- "a photo of a dog" and "a photo of a cat" are closer to each other
  than either is to "a photo of a chair"

This alignment is exactly what we want for grounded language learning.

```python
class TextEncoder:
    def __init__(self, clip_wrapper: CLIPWrapper):
        """Shares the CLIPWrapper instance with ImageEncoder — one model load."""

    def encode(self, text: str) -> torch.Tensor:
        """
        Returns: tensor of shape (512,), L2-normalized, on CPU.
        
        Preprocessing applied internally:
          - Truncate to CLIP's 77-token limit
          - Tokenize with CLIP tokenizer
        
        If text exceeds 77 tokens, it is truncated to fit.
        Long text should be summarized before encoding — see note below.
        """

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Returns: tensor of shape (N, 512), each row L2-normalized."""
```

**The 77-token limit:**
CLIP's text encoder has a hard 77-token context window. For short questions
and answers this is fine. For longer teacher answers (Stage 3-4), the answer
should be split into sentences and each sentence encoded separately.
The Baby Model receives multiple vectors per answer, one per sentence.
The Learning Loop handles this splitting — the TextEncoder just encodes
whatever string it receives.

```python
# In Learning Loop, before calling TextEncoder:
sentences = split_into_sentences(teacher_answer)   # simple sent_tokenize
vectors = [text_encoder.encode(s) for s in sentences]
# Feed vectors to Baby Model sequentially
```

---

### `VideoEncoder`

Encodes a short video clip as a sequence of frame embeddings.
Video is not treated as a special modality — it's just a sequence of images
with temporal position encodings added.

```python
class VideoEncoder:
    def __init__(
        self,
        image_encoder: ImageEncoder,
        frames_per_clip: int = 8,      # sample 8 frames from any clip
        add_position_encoding: bool = True
    ):
        """
        Reuses ImageEncoder — one model load for both image and video.
        """

    def encode(self, video_path: str) -> torch.Tensor:
        """
        Returns: tensor of shape (frames_per_clip, 512), L2-normalized per frame.
        
        Process:
          1. Load video file (mp4, mov, avi)
          2. Sample frames_per_clip frames evenly across duration
          3. Encode each frame with ImageEncoder
          4. Add sinusoidal temporal position encoding to each frame vector
          5. L2-normalize each resulting vector
        
        The temporal position encoding is a 512-dim sinusoidal vector
        scaled by 0.1 before adding — subtle temporal signal,
        doesn't overwhelm the visual content signal.
        """

    def encode_frames(
        self, frames: list[PIL.Image.Image]
    ) -> torch.Tensor:
        """
        For when frames are already extracted (e.g. from a webcam feed).
        Returns: tensor of shape (len(frames), 512).
        """
```

**Why 8 frames:**
Enough to capture the key states in a short clip (before/after an action).
More than 8 gives diminishing returns for the learning signal and increases
the number of vectors the Baby Model processes per step.
Configurable — at Stage 3 (causal learning) you may want more.

**Temporal position encoding:**
```python
def temporal_position_encoding(
    frame_index: int,
    total_frames: int,
    dim: int = 512
) -> torch.Tensor:
    """
    Sinusoidal encoding of relative position (0.0 to 1.0).
    Same principle as transformer positional encoding
    but over temporal position rather than sequence position.
    """
    position = frame_index / max(total_frames - 1, 1)  # 0.0 to 1.0
    dims = torch.arange(0, dim, 2).float()
    encoding = torch.zeros(dim)
    encoding[0::2] = torch.sin(position * 10000 ** (-dims / dim))
    encoding[1::2] = torch.cos(position * 10000 ** (-dims / dim))
    return encoding * 0.1   # scale down — subtle signal
```

---

### `TextDecoder`

Takes a vector from the Baby Model's output and produces a text response.
This is used when the human chats with the Baby Model directly.

This is the only component in the encoder/decoder that has any learned weights
beyond CLIP — it's a small generative head (a single linear layer + softmax
over a small vocabulary) that the Baby Model trains over time.

```python
class TextDecoder:
    def __init__(
        self,
        vocab_size: int = 2048,    # small vocab — common words only
        hidden_dim: int = 512      # matches embedding space
    ):
        """
        A single linear projection: R^512 → R^vocab_size
        followed by softmax.
        
        The vocabulary (vocab.py) is the 2048 most common English words
        plus the words the Baby Model has encountered in teacher answers.
        It grows as the model learns — new words from teacher answers
        are added to the vocabulary (up to a max of 8192).
        """
        self.projection = nn.Linear(hidden_dim, vocab_size)
        self.vocab = Vocabulary(max_size=8192)

    def decode(
        self,
        vector: torch.Tensor,          # (512,) output from Baby Model
        max_words: int = 30,
        temperature: float = 0.7
    ) -> str:
        """
        Autoregressive decoding over the small vocabulary.
        Generates up to max_words tokens.
        Stops at end-of-sentence token or max_words.
        
        This will produce simple, limited language at first.
        That's correct — the Baby Model is young.
        Language quality improves as the model grows.
        """

    def train_step(
        self,
        output_vector: torch.Tensor,    # (512,) Baby Model output
        target_text: str                # what it should have said
    ) -> float:
        """
        Single supervised step — trains the projection layer only.
        The Baby Model's weights are NOT updated here.
        Only the decoder projection learns from this signal.
        Returns the loss value.
        
        Called when:
          - Human sends a chat message and the model responds
          - The human rates the response (thumbs up = reinforce,
            thumbs down = train toward the correct answer)
        """
```

**Why only 2048 words to start:**
The decoder doesn't need to be expressive at Stage 0-1.
Simple answers ("it is a dog", "they are both animals") are appropriate
for an early-stage model. A small vocabulary also trains faster.
The vocabulary grows dynamically as the teacher uses new words —
any word appearing 3+ times in teacher answers gets added.

---

### `Vocabulary`

```python
class Vocabulary:
    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<START>": 1,
        "<END>": 2,
        "<UNK>": 3
    }

    def __init__(self, max_size: int = 8192):
        self.max_size = max_size
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}
        self.word_counts: dict[str, int] = {}
        self._load_base_vocab()    # loads 2048 most common English words

    def add_word(self, word: str) -> None:
        """
        Increments count for word. If count >= 3 and word not in vocab
        and vocab not full: adds to vocab.
        """

    def encode(self, text: str) -> list[int]:
        """Tokenizes text to list of ids. Unknown words → <UNK>."""

    def decode(self, ids: list[int]) -> str:
        """Converts ids back to text string."""

    def save(self, path: str) -> None:
        """Serializes to JSON — included in model checkpoints."""

    def load(self, path: str) -> None:
        """Restores from JSON."""
```

---

## Shared initialization pattern

ImageEncoder and TextEncoder share one CLIPWrapper instance.
VideoEncoder reuses ImageEncoder.
All three are instantiated once at backend startup and passed around.

```python
# In backend startup (main.py):

clip = CLIPWrapper()                          # one model load
image_encoder = ImageEncoder(clip)
text_encoder = TextEncoder(clip)
video_encoder = VideoEncoder(image_encoder)
text_decoder = TextDecoder()

# Pass to Learning Loop, Baby Model, API handlers as needed
```

Total memory for this component: ~330MB (CLIP ViT-B/32 weights).
Loaded once, stays resident. Never reloaded.

---

## Input types the system accepts

| Source | Type | Handler | Output |
|--------|------|---------|--------|
| Image file upload | bytes | `encode_bytes` | (512,) |
| Image from disk | PIL.Image | `encode` | (512,) |
| Video file | path string | `VideoEncoder.encode` | (8, 512) |
| Teacher answer | string | `TextEncoder.encode` | (512,) per sentence |
| Human chat message | string | `TextEncoder.encode` | (512,) |
| Human chat response | Baby Model output | `TextDecoder.decode` | string |

---

## What the Baby Model receives

Every input arrives as one or more 512-dim vectors.
The Baby Model does not know or care what the source was.

```
image of a dog          → one vector  (512,)
text "what is a dog?"   → one vector  (512,)
video of a ball rolling → eight vectors  (8, 512)
teacher answer (long)   → N vectors, one per sentence
```

For multi-vector inputs (video, long text), the Baby Model processes
them sequentially. The Learning Loop feeds them one at a time.
The order matters for temporal/causal learning (Stage 3).

---

## Tests

```python
# test_encoder_decoder.py

# ImageEncoder
def test_image_encoder_output_shape():
    enc = ImageEncoder(CLIPWrapper())
    img = PIL.Image.new("RGB", (224, 224), color="red")
    v = enc.encode(img)
    assert v.shape == (512,)

def test_image_encoder_output_normalized():
    enc = ImageEncoder(CLIPWrapper())
    img = PIL.Image.new("RGB", (224, 224))
    v = enc.encode(img)
    assert abs(torch.norm(v).item() - 1.0) < 1e-5

# TextEncoder
def test_text_encoder_output_shape():
    enc = TextEncoder(CLIPWrapper())
    v = enc.encode("a photo of a dog")
    assert v.shape == (512,)

def test_text_encoder_output_normalized():
    enc = TextEncoder(CLIPWrapper())
    v = enc.encode("hello world")
    assert abs(torch.norm(v).item() - 1.0) < 1e-5

def test_clip_alignment():
    """Image and matching text should be more similar than mismatched."""
    clip = CLIPWrapper()
    img_enc = ImageEncoder(clip)
    txt_enc = TextEncoder(clip)

    dog_image = PIL.Image.open("tests/fixtures/dog.jpg")
    v_dog_img = img_enc.encode(dog_image)
    v_dog_txt = txt_enc.encode("a photo of a dog")
    v_chair_txt = txt_enc.encode("a photo of a chair")

    sim_correct = torch.dot(v_dog_img, v_dog_txt).item()
    sim_wrong = torch.dot(v_dog_img, v_chair_txt).item()
    assert sim_correct > sim_wrong

# VideoEncoder
def test_video_encoder_output_shape():
    clip = CLIPWrapper()
    img_enc = ImageEncoder(clip)
    vid_enc = VideoEncoder(img_enc, frames_per_clip=8)
    frames = [PIL.Image.new("RGB", (224, 224)) for _ in range(16)]
    v = vid_enc.encode_frames(frames)
    assert v.shape == (8, 512)   # always returns frames_per_clip frames

def test_video_encoder_temporal_ordering():
    """Earlier frames should have different encoding than later frames."""
    clip = CLIPWrapper()
    img_enc = ImageEncoder(clip)
    vid_enc = VideoEncoder(img_enc, frames_per_clip=4)
    # Same image content, different temporal positions
    frame = PIL.Image.new("RGB", (224, 224), color="blue")
    frames = [frame] * 8
    v = vid_enc.encode_frames(frames)
    # Vectors should differ because of position encoding
    assert not torch.allclose(v[0], v[-1])

# Vocabulary
def test_vocabulary_special_tokens():
    vocab = Vocabulary()
    assert vocab.encode("<START>") == [1]
    assert vocab.encode("<END>") == [2]

def test_vocabulary_grows():
    vocab = Vocabulary(max_size=100)
    for _ in range(3):
        vocab.add_word("serendipity")
    assert "serendipity" in vocab.word_to_id

def test_vocabulary_max_size_respected():
    vocab = Vocabulary(max_size=10)
    for i in range(20):
        for _ in range(3):
            vocab.add_word(f"word_{i}")
    assert len(vocab.word_to_id) <= 10
```

---

## Hard parts

**CLIP's 77-token limit cuts real answers.**
At Stage 3 and 4, teacher answers will regularly exceed 77 tokens.
Sentence splitting before encoding is necessary — but sentence boundary
detection is not trivial. Use `nltk.sent_tokenize` (available, reliable)
rather than splitting on periods (breaks on "Dr.", "U.S.", decimals).
Add `nltk` to backend dependencies and download the punkt tokenizer
during `start.sh` setup.

**First CLIP load takes 10-30 seconds.**
The 330MB model downloads on first run and loads into memory.
Backend startup should show a clear progress indicator rather than
appearing to hang. After the first run it's cached locally — subsequent
starts take 2-4 seconds.

**MLX and PyTorch tensor interop.**
MLX tensors cannot be directly passed to PyTorch operations.
Every MLX result must go through numpy before arriving at PyTorch:
```python
mlx_result = mx.array(...)
# WRONG:
torch_tensor = torch.from_numpy(mlx_result)  # MLX arrays are not numpy
# RIGHT:
torch_tensor = torch.from_numpy(np.array(mlx_result))
```
This conversion is fast (shared memory on M1 — no copy needed)
but the explicit `np.array()` call is required.

**TextDecoder quality at early stages will be poor.**
The decoder head is a single linear layer over 2048 words.
Early in training it will produce grammatically broken, repetitive responses.
This is correct behavior, not a bug. Resist the urge to make it
"better" by adding more parameters — the point is that the Baby Model's
*output quality reflects its developmental stage*. A richer decoder
would paper over that signal.

**Video file reading dependency.**
`VideoEncoder.encode(video_path)` needs a video reading library.
Use `opencv-python-headless` (not `opencv-python` — the headless version
has no GUI dependencies, smaller install, M1 compatible):
```bash
pip install opencv-python-headless
```
Frame extraction:
```python
import cv2
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
indices = np.linspace(0, total_frames - 1, frames_per_clip, dtype=int)
frames = []
for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(PIL.Image.fromarray(frame_rgb))
cap.release()
```

---

## M1-specific notes

**Memory layout.**
CLIP ViT-B/32 in float32: ~330MB.
In MLX on M1, this lives in unified memory — shared between CPU and GPU.
No VRAM separate from RAM. No data transfer between devices.
The `encode` call uses the Neural Engine for matrix ops automatically.

**MLX install:**
```bash
pip install mlx
pip install git+https://github.com/ml-explore/mlx-examples  # for CLIP
```
MLX requires macOS 13.5+ and Apple Silicon (M1/M2/M3).
The system should check for this in `start.sh` and fall back to
PyTorch MPS if MLX is not available, with a warning that
image encoding will be slower.

**Batch size on M1.**
For `encode_batch`, optimal batch size on M1 is 16-32 images.
Larger batches don't fit efficiently in the Neural Engine's tile size.
The batch methods default to processing in chunks of 16 internally.

**Float precision.**
MLX defaults to float32. CLIP was trained in float16 or float32.
Keep everything in float32 throughout this component.
The Baby Model may use float32 or bfloat16 internally — the conversion
happens at the boundary, not inside the encoder.
