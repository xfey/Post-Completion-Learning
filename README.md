## Post-Completion Learning for Language Models

This is the official code for post-completion learning.

Paper link: [arXiv](https://arxiv.org/abs/2507.20252)

#### Code

Our code is based on [`open-r1`](https://github.com/huggingface/open-r1), with our customized `Trainer` for mixed SFT+GRPO training. Some other updates focus on the white-box RL (reward function design) and post-completion training (replacement of <eos>, parallel SFT, etc.)

The code repository and training data will be fully open source in the near future.
