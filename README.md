# Diffusion Policy on Push-T

A from-scratch implementation of **Diffusion Policy** trained on the **Push-T** imitation-learning dataset. It builds on an earlier [DDPM image-generation project](https://github.com/andrew-s-h/diffusion): the same denoising machinery that generated CelebA faces is used here to generate *robot actions*.

The project runs in two stages:

1. **Keypoints (current).** The policy is told exactly where the T-block is through 8 keypoints on its outline. There is no vision encoder, so the work focuses entirely on the diffusion part, and training is fast.
2. **Images (next).** The keypoints are replaced with camera frames and a vision encoder, which turns it into a full visuomotor policy. The diffusion core, U-Net, normalization, and evaluation carry over. Only the observation encoder changes.

Running the keypoint stage first means bugs in the diffusion part can't be confused with bugs in the vision part. It also gives a reference score for the image stage to aim for.

## The Task

Push-T is a 2D simulated manipulation task. A circular agent (think of it as a robot fingertip) moves around a flat table. A T-shaped block sits on the table, and a fixed green T-shaped outline marks the goal. The agent has to push the block until it lines up with the goal outline.

The task is hard for a simple policy for two reasons:

- **Contact-rich.** The agent can only move the block by pushing it, so it has to approach from the correct side to slide or rotate it.
- **Multimodal.** There is often more than one good move. You could go around the block to the left or to the right. A policy that averages the two ends up driving straight into the block. Diffusion Policy is built to handle this.

An episode succeeds when the block covers at least **95%** of the goal region.

## Key Terms: State, Action, Keypoints, and Simulator State

These terms are easy to mix up, and they determine what the model sees and what it predicts. Two of them share a name: the dataset column `observation.environment_state` holds the **keypoints**, which is not the same thing as the 5-number **simulator state**.

| Term | What it is | Shape | Seen by our policy? |
|---|---|---|---|
| **State** (`observation.state`) | Where the agent is right now: its `(x, y)` position | 2 | ✅ Yes (input, both stages) |
| **Keypoints** (`observation.environment_state`) | 8 `(x, y)` points on the T-block's outline, which give its position and rotation | 16 | ✅ Stage 1 input · ❌ Stage 2 |
| **Image** (`observation.image`) | A 96×96 RGB render of the whole scene | 96×96×3 | ❌ Stage 1 · ✅ Stage 2 input |
| **Action** (`action`) | Where the agent is told to move next: a target `(x, y)` position | 2 | 🎯 **Predicted** (output) |
| **Simulator state** | The full ground truth of the scene: agent position plus the block's position and rotation | 5 | ❌ No (the simulator uses it) |

**State is about the agent.** It answers "where am I?" It says nothing about the block.

**Action is a target position, not a velocity.** If the action is `(300, 250)`, the agent is pulled toward that point. In a smooth demonstration the action is usually close to the current state, a little way ahead in the direction of motion.

**Keypoints are about the block.** They give the block's pose as 8 points, which is easier for a network to use than a raw angle, because an angle jumps from 2π back to 0. In stage 1 the policy gets them directly. In stage 2 it doesn't, and has to work out where the block is from the **image**. That is what makes it a *visuomotor* policy: it maps pixels to motion.

**Simulator state is about the whole world.** It is the 5-number vector `[agent_x, agent_y, block_x, block_y, block_angle]`. The simulator uses it to step the physics, to score how well the block covers the goal, and to reset a scene to a known configuration (`reset_to_state`). Our policy never receives it.

All positions are in workspace coordinates in the range `[0, 512]`. The block angle is in radians, `[0, 2π]`.

**What is being predicted, in one sentence:** given the last couple of observations (block keypoints in stage 1, camera frames in stage 2) and agent positions, predict the next several target positions the agent should move to.

## Dataset

Both stages use the same 206 human demonstrations from the Hugging Face Hub, in LeRobot format. They differ only in how the block is observed:

| Stage | Dataset | Block observation | License |
|---|---|---|---|
| 1 (current) | [`lerobot/pusht_keypoints`](https://huggingface.co/datasets/lerobot/pusht_keypoints) | `observation.environment_state` (16 keypoint values) | Apache-2.0 |
| 2 | [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht) | `observation.image` (AV1 video) | MIT |

| Property | Value |
|---|---|
| Episodes | 206 human demonstrations |
| Frames | 25,650 total |
| Frame rate | 10 fps |
| Split | Single `train` split |

Each row is one timestep of one episode:

| Column | Type | Description |
|---|---|---|
| `observation.environment_state` | float32 [16] | Stage 1 only: 8 block keypoints, `(x, y)` each |
| `observation.image` | video frame, 96×96×3 | Stage 2 only: rendered scene (stored as AV1 video, decoded per frame) |
| `observation.state` | float32 [2] | Agent `(x, y)` |
| `action` | float32 [2] | Target `(x, y)` commanded at this step |
| `episode_index` | int64 | Which demonstration this frame belongs to |
| `frame_index` | int64 | Position of the frame within its episode |
| `timestamp` | float32 | Time in seconds within the episode |
| `next.reward` | float32 | Goal coverage after this step, scaled so that 95% coverage = 1.0 (`clip(coverage / 0.95, 0, 1)`) |
| `next.done` | bool | Whether the episode ended after this step |
| `next.success` | bool | Whether the task was solved after this step |
| `index`, `task_index` | int64 | Global frame index and task ID (a single task here) |

People recorded the demonstrations by controlling the agent in the simulator. Different demonstrations handle similar situations in different ways. That variety is the multimodality the model has to capture.

## The Model: Diffusion Policy

A standard policy looks at the scene and outputs one action. Diffusion Policy instead **generates a short sequence of future actions** with the same denoising process that diffusion models use to generate images.

Here is how it works:

1. **Look.** Take the last 2 observations and the last 2 agent positions. Together they form the *conditioning*: a summary of "what's going on right now."
   - **Stage 1:** the block keypoints and agent positions go in directly, as 2 × 18 normalized numbers.
   - **Stage 2:** a small CNN (a ResNet-18) turns each camera frame into a feature vector, which takes the place of the keypoints.

2. **Start from noise.** Create a random sequence of 16 future `(x, y)` actions. At this point it is pure noise, a meaningless scribble of points.

3. **Denoise.** A 1D U-Net that runs over the time axis of the action sequence looks at the noisy actions and the conditioning, then removes a little of the noise. Repeat this many times. The scribble gradually turns into a smooth, sensible trajectory. This is the same idea as denoising static into a face, except the "image" is a path through space and the denoising is guided by what the agent observes.

4. **Act, then re-plan.** Run the first 8 actions of the trajectory in the environment, then take new observations and generate a fresh plan. Committing to several steps at a time keeps motion smooth. Re-planning keeps the policy reactive when the block doesn't move as expected.

**Training** uses the DDPM recipe: take a real action sequence from the dataset, add a random amount of noise, and train the network to predict that noise, given the observations.

**Why this handles multimodality:** each run starts from different random noise, so the sampler can land on *either* "go left" *or* "go right" as a coherent plan instead of averaging them into a bad compromise.

The horizon values above (2 observed steps, 16 predicted, 8 executed) are the original paper's Push-T defaults and the starting point for this project.

## Building on the DDPM Project

The [earlier project](https://github.com/andrew-s-h/diffusion) was a from-scratch DDPM (ε-prediction, fixed posterior variance) trained on 64×64 CelebA faces. The diffusion core doesn't care whether it's denoising pixels or action sequences, so most of it carries over. What changes is the *data* being denoised, and the denoising is now *conditioned* on what the agent observes.

**Carried over largely unchanged**
- **Gaussian diffusion core:** forward noising (corrupting clean data in one step), the reverse posterior, the reverse sampling loop, and the simple noise-prediction loss
- **x̂₀ clamping to `[-1, 1]`:** still valid, because actions are normalized into the same range that pixels were
- **Sinusoidal timestep embedding:** tells the network how noisy its input is
- **EMA weights:** a slow-moving average of the weights, used for evaluation
- **Training-loop structure:** batching, optimizer, checkpointing, and logging
- **Preprocess-once data pattern:** decode the dataset to disk once, then train from memory-mapped arrays. This matters even more in stage 2, because the frames are stored as compressed video. Stage 1 is small enough to fit in memory straight from the parquet files.

**Adapted**
- **Data shape:** instead of a 64×64×3 image, each sample is a 16×2 action sequence (16 timesteps of `(x, y)`)
- **U-Net:** the 2D image U-Net becomes a 1D U-Net that convolves along the time axis of the action sequence
- **Noise schedule:** the DDPM project used a linear schedule with 1000 steps. Diffusion Policy uses a cosine schedule with far fewer steps (100 in the paper). Sampling runs inside a control loop, so every step costs reaction time.
- **Normalization:** pixels were mapped to `[-1, 1]` with a fixed scale. States, keypoints, and actions are mapped to `[-1, 1]` using per-dimension min/max values computed from the dataset (`scripts/inspect_data.py` writes these to `norm_stats.json`).

**New**
- **Observation conditioning:** the faces model was unconditional. Here, the observations (keypoints in stage 1, image features in stage 2) and agent positions are injected into every U-Net block, alongside the timestep embedding, to steer the denoising.
- **Vision encoder (stage 2):** a ResNet-18 that turns each camera frame into a feature vector
- **Episode windowing:** cutting each demonstration into (observation history, future action chunk) training pairs
- **Rollout evaluation:** FID on generated faces is replaced by task success in simulation (see [Simulation](#simulation))

## Simulation

We don't generate any new training data. The 206 demonstrations are the entire training set. The simulator is used to **evaluate** the trained policy by letting it attempt the task on its own.

We use [`gym-pusht`](https://github.com/huggingface/gym-pusht), Hugging Face's Gymnasium version of the same environment the demonstrations were recorded in. It is a 2D physics simulation: each action sets a target position, a controller pulls the agent toward it, and the agent collides with and pushes the block. It can return the same observations the model was trained on: block keypoints (`obs_type="environment_state_agent_pos"`, stage 1) or the 96×96 rendered view (stage 2). That way, what the policy sees at test time matches what it saw in training.

Each evaluation episode works like this:

1. **Reset** the environment with a seed. The seed sets the T-block's random starting position and rotation.
2. **Observe** the keypoints or camera frame and the agent position, and keep the last 2 of each.
3. **Plan** a 16-step action sequence with the trained policy.
4. **Execute** the first 8 actions one at a time. The simulator steps the physics and reports goal coverage after each one.
5. **Repeat** from step 2 until the block reaches 95% coverage or the episode hits its 300-step limit.

Across many seeded episodes we report:
- **Success rate:** the fraction of episodes that reach 95% coverage
- **Average max coverage:** the best coverage reached in each episode, averaged across episodes. It gives partial credit, so it shows progress before the success rate does.

The same seeds are used for every checkpoint, so results are directly comparable. Rollout videos are saved so failures can be inspected by eye.

## Getting Started

Requires Python 3.13. Run all commands from the repo root, because the scripts use relative data paths.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download the stage-1 dataset (about 2 MB) into data/
hf download lerobot/pusht_keypoints --repo-type dataset --local-dir data/pusht_keypoints
```

### Phase 0: data checks

Before writing any model code, these scripts confirm the data means what we think it means.

| Script | What it checks |
|---|---|
| `scripts/inspect_data.py` | Episode and frame indexing are contiguous; prints value ranges; writes `norm_stats.json` |
| `scripts/plot_episode.py <ep>` | Plots state against action over time, and measures how many frames the pusher lags behind the cursor |
| `scripts/replay_episode.py <ep...>` | Replays a demo's recorded actions in `gym-pusht` and compares the simulated pusher path with the recorded one |

The replay matches the recorded pusher positions with **0.00 px error**. This confirms that actions are target positions, that action `t` produces state `t+1`, and that `gym-pusht` is the same simulator the demonstrations were recorded in.

## Project Scope

**In scope**
- Loading and preprocessing `lerobot/pusht_keypoints` (stage 1) and `lerobot/pusht` (stage 2) into training windows (observation history paired with future action chunks)
- Normalizing states, keypoints, and actions
- Conditional 1D temporal U-Net, DDPM noise schedule and sampler, and (stage 2) a vision encoder
- Training loop with EMA weights
- Rollout evaluation in `gym-pusht`, reporting success rate and goal coverage

**Out of scope (for now)**
- Real-robot deployment
- The transformer-based Diffusion Policy variant
- Faster samplers (e.g., DDIM) beyond a possible later experiment
- Other LeRobot tasks

## Attribution

This project reimplements published work and uses public data and tools. All credit for the method, task, and dataset goes to the original authors.

- **Diffusion Policy.** Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023; extended version in *The International Journal of Robotics Research*, 2024. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) · [Project page](https://diffusion-policy.cs.columbia.edu/)
- **Push-T task.** Adapted by the Diffusion Policy authors from the task in Pete Florence et al., *Implicit Behavioral Cloning*, CoRL 2021.
- **Datasets.** [`lerobot/pusht_keypoints`](https://huggingface.co/datasets/lerobot/pusht_keypoints) (Apache-2.0) and [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht) (MIT), released by Hugging Face LeRobot and converted from the original Diffusion Policy demonstrations.
- **Simulator.** [`gym-pusht`](https://github.com/huggingface/gym-pusht), Hugging Face, Apache-2.0 license, adapted from Diffusion Policy.
- **DDPM.** Jonathan Ho, Ajay Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. This project's diffusion core is reused from [andrew-s-h/diffusion](https://github.com/andrew-s-h/diffusion).