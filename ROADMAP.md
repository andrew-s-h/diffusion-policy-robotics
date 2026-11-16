# Roadmap

Each step is small and has a check that tells you it's done.

## Phase 0: Data checks ✅
- `inspect_data.py`: indexing and value ranges, writes `norm_stats.json`
- `plot_episode.py`: the pusher trails the cursor
- `replay_episode.py`: the simulator reproduces the demos (0.00 px)

Note: human demos end at 86–97% coverage, and only 7 of 206 cross the 95% success line. Judge the model mainly on coverage, not success rate.

## Phase 1: Normalization ✅
- `normalize(x, stats)` maps values to [-1, 1], and `unnormalize` reverses it
- Done when: a round trip on the whole dataset gives back the original values

## Phase 2: Training windows
- Cut each episode into (last 2 observations, next 16 actions) pairs. At the start and end of an episode, repeat the first or last frame to fill the window.
- Wrap it in a PyTorch `Dataset`
- Done when: the shapes are `(2, 18)` and `(16, 2)`, and plotting one window shows the actions continuing on from the pusher's path

## Phase 3: Diffusion core (ported from the DDPM project)
- Cosine noise schedule with 100 steps, forward noising, noise-prediction loss, reverse sampler
- Done when: it learns a simple 2D toy distribution with two separate clusters and samples from both

## Phase 4: The model
- A 1D U-Net over the 16-step time axis, with the observations and timestep fed in through FiLM
- Done when: it can overfit one batch, with loss going to about 0

## Phase 5: Training
- Full training loop with EMA, checkpoints and a loss curve. Short runs on the Mac, full runs on the 5070 Ti.
- Done when: the loss levels off and sampled action chunks look smooth

## Phase 6: Evaluation
- Rollout loop: observe → plan 16 actions → run 8 → repeat, for up to 300 steps
- Fixed seeds (e.g. 50). Report average max coverage first and success rate second, and save videos.
- Done when: you have one number per checkpoint and can watch the failures

## Phase 7: Experiments (optional)
- A plain MLP baseline that predicts one action, to show the averaging problem in practice
- Changing the horizons (2 / 16 / 8), and faster sampling with DDIM

## Phase 8: Stage 2, images
- Switch to `lerobot/pusht`, add a ResNet-18 encoder, and compare the scores with stage 1
