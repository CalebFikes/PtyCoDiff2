Debug Summary for Copilot on new machine

Short context:
- Repository: PtyCoDiff2 (workspace root). I debugged exploding activations in conditional sampling and fixed several things:
  - `att_scale` is now treated as a static hyperparameter (passed into `create_complexUnet`).
  - Weight-standardization (WS) is skipped for 1x1 kernels to avoid amplifying near-identity mix/proj kernels.
  - Removed on-the-fly legacy `.pkl` converter: loader `load_checkpoint_params` now requires training-saved `.npz` (treedef + param leaves + metadata).
  - Sampler schedule aligned with training (cosine schedule with s0=0.008) and `score_apply` converts model eps->score using same sigma.
  - `scripts/run_conditional_viz.py` now maps dataset images to [-1,1], has `--diagnostics` and `--trace_layers`, and a `--force_zero_att_mix` debug flag.

Files to inspect (changed):
- diffusion/complex_unet.py
- diffusion/model.py
- diffusion/sampler.py
- training/train.py
- scripts/run_conditional_viz.py

Key runtime notes & caveats:
- Training and sampling use JAX. Use a machine with sufficient GPU memory. Earlier run failed with CUDA OOM during autotuning / cuDNN conv profiling.
- If you hit cuDNN autotune errors, set XLA_FLAGS to relax strict conv algorithm picker:
  export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
  or run the command prefixed with that env var.
- Checkpoints must be the NPZ format produced by `training.save_pytree_npz`; legacy .pkl conversion is disabled.

Exact commands I used (copy/paste):

# Run a 1-epoch smoke training (uses conda env Python in this workspace)
PYTHONPATH=/local/scratch/cfikes/PtyCoDiff2 /local/scratch/cfikes/PtyCoDiff2/conda-env/bin/python \
  training/train.py --dataset_dir datasets/mnist --epochs 1 --batch_size 64 \
  --out_weights runs/test_ckpt.npz --base_ch 16 --att_scale 0 --n_t 32 --diagnostics

# Run conditional viz on the produced NPZ (example)
PYTHONPATH=/local/scratch/cfikes/PtyCoDiff2 /local/scratch/cfikes/PtyCoDiff2/conda-env/bin/python \
  scripts/run_conditional_viz.py --ckpt runs/test_ckpt.npz --mnist datasets/mnist/train.npz \
  --outdir runs/conditional_viz_test --J 1000 --P 8 --n_steps 128 --measurement_weight 1.0 --diagnostics

# If CUDA OOM / cuDNN autotune errors appear, try (one-liner):
XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false PYTHONPATH=/local/scratch/cfikes/PtyCoDiff2 \
  /local/scratch/cfikes/PtyCoDiff2/conda-env/bin/python training/train.py ...

What to check first on a fresh machine:
- Python environment: activate the conda env or use the full path shown above. Ensure `jax` and `jaxlib` are installed with GPU support matching CUDA on the host.
- GPU memory: allocate at least one GPU with >=12–16GB for `base_ch=32`; reduce `--base_ch` to 16 or 8 for quick smoke runs.
- Run the training command above and save the produced `runs/test_ckpt.npz`.
- Then run the viz command to validate sampling; use `--diagnostics` and `--trace_layers` if outputs look incorrect.

If you want full legacy `.pkl` converter support, I can add a careful converter preserving complex channel stacking semantics, but current policy is to require NPZ for correctness.

Contact notes for the new Copilot instance:
- I expect you to run the training command first (smoke), then the viz command with `--ckpt` pointing to the saved NPZ.
- If you hit GPU memory errors, reduce `--base_ch` or `--batch_size`, and/or add `XLA_FLAGS` as shown.

End of prompt.
