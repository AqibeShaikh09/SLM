# Transformer on TinyStories

A from-scratch PyTorch implementation of the encoder-decoder Transformer from
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), following the structure of
[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), trained to continue children's stories
from the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

Everything is in [`attention_blog_tinystories_optimized.ipynb`](attention_blog_tinystories_optimized.ipynb).

## How it works

Each story is tokenized and cut into 32-token chunks. The encoder reads three consecutive chunks (96 tokens) and the
decoder predicts the next chunk (32 tokens), sliding across the story one chunk at a time. Because every window has the
same shape, batches need no padding. To generate, the model repeatedly appends a predicted chunk and slides the window forward.

| Component | Details |
|---|---|
| Tokenizer | Byte-level BPE, 8,000 tokens, trained on TinyStories |
| Architecture | 4 encoder + 4 decoder layers (pre-norm), `d_model=384`, 6 heads, `d_ff=1408`, dropout 0.1 |
| Parameters | 25.0M |
| Optimizer | AdamW (β = 0.9, 0.98), weight decay 0.01, gradient clipping 1.0 |
| LR schedule | Noam: linear warmup for 4,000 steps, then inverse-sqrt decay |
| Decoding | Greedy or top-k sampling with temperature |
| Extras | Mixed-precision (AMP) training, best/last checkpointing, streaming data loading |

## Results

5 epochs on a single NVIDIA RTX 3090 (training capped at 300k stories, validation at 50k, batch size 64,
roughly 15.7k steps per epoch at about 20k tokens/s):

| Epoch | Train loss | Train ppl | Val loss | Val ppl |
|---|---|---|---|---|
| 1 | 2.730 | 15.33 | 2.110 | 8.25 |
| 2 | 2.089 | 8.07 | 1.949 | 7.02 |
| 3 | 1.972 | 7.18 | 1.880 | 6.55 |
| 4 | 1.907 | 6.73 | 1.835 | 6.27 |
| 5 | 1.863 | 6.45 | **1.807** | **6.09** |

Validation loss improved every epoch, so more training would likely help.

Sample continuation (top-k = 50, temperature = 0.9) for the prompt *"Once upon a time, there was"*:

> Once upon a time, there was was very messy, and they had to clean their room.
>
> Tom looked at the wall and saw that the table had fallen on. He was sad because room and the floor was messy.
>
> Her mom heard her cry and came to the table. She saw the mess making a mess and said, "What and mess.
>
> From that day on, Tom learned to be more careful and clean up the mess. He loved to help in his room. They played
> together every day and had lots of fun. And from that day on, Tom always listened to his mom and never touched a messy
> wall again. The end.

The model picks up TinyStories' style and story arc. It stumbles where the 32-token chunks meet, which is expected
because each chunk is decoded separately.

## Requirements

Python 3.10+, PyTorch, `datasets`, `tokenizers`, `tqdm`. A CUDA GPU is strongly recommended.

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017
- Rush et al., [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- Eldan & Li, [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759), 2023
