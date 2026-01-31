import os
import time
from typing import List

import lightning as L
import prodigyopt
import torch
import wandb
import yaml
from diffusers.models.controlnets.controlnet import zero_module
from diffusers.pipelines import FluxPipeline
from lightning.pytorch.callbacks import ModelCheckpoint
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader

from ..pipeline.flux_refany3d import encode_images, transformer_forward


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("REFANY3D_CONFIG")
    assert config_path is not None, "Please set the REFANY3D_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_group_mask(n_branch, n_condition):
    group_mask = torch.zeros((n_branch, n_branch), dtype=bool)
    group_mask[0, :] = True
    group_mask[1, :] = True
    group_mask[2, 1:] = True

    group_mask[3 : n_condition + 3, 3 : n_condition + 3] = torch.eye(n_condition, dtype=bool)
    group_mask[3 : n_condition + 3, n_condition + 3 : 2 * n_condition + 3] = torch.eye(n_condition, dtype=bool)
    group_mask[n_condition + 3 : 2 * n_condition + 3, 3 : n_condition + 3] = torch.eye(n_condition, dtype=bool)
    group_mask[n_condition + 3 : 2 * n_condition + 3, n_condition + 3 : 2 * n_condition + 3] = torch.eye(
        n_condition, dtype=bool
    )

    group_mask[3:, :3] = True  # Enable the attention from condition branches to image branch and text branch
    return group_mask


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


class RefAny3DModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        adapter_names: List[str] = [None, None, "default"],
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        aux_token: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(flux_pipe_id, torch_dtype=dtype).to(device)
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()
        self.adapter_names = adapter_names
        self.adapter_set = set([each for each in adapter_names if each is not None and "+" not in each])

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)

        # set group_mask
        self.model_config["group_mask"] = get_group_mask(
            n_branch=len(self.adapter_names),
            n_condition=(len(self.adapter_names) - 3) // 2,
        )

        self.transformer.aux_token = aux_token
        if aux_token:
            self.transformer.domain_embedder = torch.nn.Sequential(
                torch.nn.Embedding(
                    num_embeddings=2,
                    embedding_dim=256,
                ),
                torch.nn.Linear(256, self.transformer.inner_dim, bias=True),
                torch.nn.SiLU(),
                zero_module(torch.nn.Linear(self.transformer.inner_dim, self.transformer.inner_dim, bias=True)),
            )
            self.transformer.domain_embedder = self.transformer.domain_embedder.to(device).to(dtype)
            self.transformer.domain_embedder.requires_grad_(True)
            self.lora_layers.extend(
                list(filter(lambda p: p.requires_grad, self.transformer.domain_embedder.parameters()))
            )


        self.to(device).to(dtype)

    def init_lora(self, lora_path: dict, lora_config: dict):
        assert lora_path or lora_config
        for adapter_name in self.adapter_set:
            self.transformer.add_adapter(LoraConfig(**lora_config), adapter_name=adapter_name)

        if lora_path:
            from safetensors.torch import load_file

            lora_weights = {}
            for name, path in lora_path.items():
                lora_weights[name] = {}
                state_dict = load_file(path)
                for key in state_dict:
                    new_key = key.replace("transformer.", "").replace(".weight", f".{name}.weight")
                    lora_weights[name][new_key] = state_dict[key]
                self.transformer.load_state_dict(lora_weights[name], strict=False)

        self.transformer.set_adapter(list(self.adapter_set))

        # TODO: Check if this is correct (p.requires_grad)
        name_list = []
        for name, param in self.transformer.named_parameters():
            if any(adapter in name for adapter in self.adapter_set):
                param.requires_grad_(True)
                name_list.append(name)

        lora_layers = filter(lambda p: p.requires_grad, self.transformer.parameters())
        # print trainable params by adapter
        adapter_params = {}
        total_trainable_params = 0

        for name, param in self.transformer.named_parameters():
            if param.requires_grad:
                # Extract adapter name (usually contained in parameter name)
                adapter_name = "unknown"
                for adapter in self.adapter_set:
                    if adapter in name:
                        adapter_name = adapter
                        break

                if adapter_name not in adapter_params:
                    adapter_params[adapter_name] = {"count": 0, "params": []}

                param_count = param.numel()
                adapter_params[adapter_name]["count"] += param_count
                adapter_params[adapter_name]["params"].append((name, param.shape, param_count))
                total_trainable_params += param_count

        # Print parameter statistics for each adapter
        print("=" * 60)
        print("Trainable Parameters Statistics:")
        print("=" * 60)

        for adapter_name, info in adapter_params.items():
            print(f"\nAdapter: {adapter_name}")
            print(f"Total parameters: {info['count']:,}")
            print("Parameter details:")
            # for param_name, shape, count in info["params"]:
            #     print(f"  - {param_name}: {shape} ({count:,} parameters)")

        print(f"\nTotal trainable parameters: {total_trainable_params:,}")
        print("=" * 60)

        return list(lora_layers)

    def save_lora(self, path: str):
        for adapter_name in self.adapter_set:
            FluxPipeline.save_lora_weights(
                save_directory=path,
                weight_name=f"{adapter_name}.safetensors",
                transformer_lora_layers=get_peft_model_state_dict(self.transformer, adapter_name=adapter_name),
                safe_serialization=True,
            )

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError("Optimizer not implemented.")
        return optimizer

    def training_step(self, batch, batch_idx):
        rgb_lambda = 1
        coord_lambda = 1


        imgs, prompts = batch["image"], batch["description"]
        image_latent_mask = batch.get("image_latent_mask", None)

        coord_imgs, coord_prompts = batch["coord_image"], batch["coord_description"]
        coord_position_deltas = batch.get("coord_image_position_delta", [[0, 0]])

        # Get the conditions and position deltas from the batch
        conditions, position_deltas, position_scales, latent_masks = [], [], [], []
        for i in range(1000):
            if f"condition_{i}" not in batch:
                break
            conditions.append(batch[f"condition_{i}"])
            position_deltas.append(batch.get(f"position_delta_{i}", [[0, 0]]))
            position_scales.append(batch.get(f"position_scale_{i}", [1.0])[0])
            latent_masks.append(batch.get(f"condition_latent_mask_{i}", None))

        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs)

            coord_x_0, coord_img_ids = encode_images(self.flux_pipe, coord_imgs)
            coord_img_ids[:, 1] += coord_position_deltas[0, 0]
            coord_img_ids[:, 2] += coord_position_deltas[0, 1]

            # Prepare text input
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.flux_pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.flux_pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get("max_sequence_length", 512),
                lora_scale=None,
            )

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
            if image_latent_mask is not None:
                x_0 = x_0[:, image_latent_mask[0]]
                x_1 = x_1[:, image_latent_mask[0]]
                x_t = x_t[:, image_latent_mask[0]]
                img_ids = img_ids[image_latent_mask[0]]

            # Prepare t and x_t
            coord_x_1 = torch.randn_like(coord_x_0).to(self.device)
            coord_x_t = ((1 - t_) * coord_x_0 + t_ * coord_x_1).to(self.dtype)
            if image_latent_mask is not None:
                coord_x_0 = coord_x_0[:, image_latent_mask[0]]
                coord_x_1 = coord_x_1[:, image_latent_mask[0]]
                coord_x_t = coord_x_t[:, image_latent_mask[0]]
                coord_img_ids = coord_img_ids[image_latent_mask[0]]

            # Prepare conditions
            condition_latents, condition_ids = [], []
            for cond, p_delta, p_scale, latent_mask in zip(conditions, position_deltas, position_scales, latent_masks):
                # Prepare conditions
                c_latents, c_ids = encode_images(self.flux_pipe, cond)
                # Scale the position
                if p_scale != 1.0:
                    scale_bias = (p_scale - 1.0) / 2
                    c_ids[:, 1:] *= p_scale
                    c_ids[:, 1:] += scale_bias
                # Add position delta 
                c_ids[:, 1] += p_delta[0][0]
                c_ids[:, 2] += p_delta[0][1]
                if len(p_delta) > 1:
                    print("Warning: only the first position delta is used.")
                # Append to the list
                if latent_mask is not None:
                    c_latents, c_ids = c_latents[latent_mask], c_ids[latent_mask[0]]
                condition_latents.append(c_latents)
                condition_ids.append(c_ids)

            # Prepare guidance
            guidance = torch.ones_like(t).to(self.device) if self.transformer.config.guidance_embeds else None

        branch_n = 3 + len(conditions)
        group_mask = self.model_config["group_mask"].detach().clone().to(self.device, dtype=torch.bool)


        # Forward pass
        transformer_out = transformer_forward(
            self.transformer,
            image_features=[x_t, coord_x_t, *(condition_latents)],
            text_features=[prompt_embeds],
            img_ids=[img_ids, coord_img_ids, *(condition_ids)],
            txt_ids=[text_ids],
            # There are three timesteps for the three branches
            # (text, image, and the condition)
            timesteps=[t, t, t] + [torch.zeros_like(t)] * len(conditions),
            # Same as above
            pooled_projections=[pooled_prompt_embeds] * branch_n,
            guidances=[guidance] * branch_n,
            # The LoRA adapter names of each branch
            adapters=self.adapter_names,
            return_dict=False,
            group_mask=group_mask,
        )
        pred = transformer_out[0]
        coord_pred = transformer_out[1]

        # Compute loss
        rgb_step_loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        coord_step_loss = torch.nn.functional.mse_loss(coord_pred, (coord_x_1 - coord_x_0), reduction="mean")

        step_loss = rgb_lambda * rgb_step_loss + coord_lambda * coord_step_loss

        self.last_t = t.mean().item()

        self.log_loss = (
            step_loss.item() if not hasattr(self, "log_loss") else self.log_loss * 0.95 + step_loss.item() * 0.05
        )

        return {
            "loss": step_loss,
            "rgb_loss": rgb_step_loss.detach(),
            "coord_loss": coord_step_loss.detach(),
        }

    def generate_a_sample(self):
        raise NotImplementedError("Generate a sample not implemented.")


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}, test_function=None):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = wandb is not None and os.environ.get("WANDB_API_KEY") is not None

        self.total_steps = 0
        self.test_function = test_function

    def on_train_start(self, trainer, pl_module) -> None:
        """Called when the train begins."""
        self.total_steps = pl_module.global_step if hasattr(pl_module, "global_step") else 0

        accumulate_grad_batches = self.training_config.get("accumulate_grad_batches", 1)
        self.total_steps *= accumulate_grad_batches

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                # "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t

            rgb_loss_value = outputs["rgb_loss"].item() * trainer.accumulate_grad_batches
            report_dict["rgb_loss"] = rgb_loss_value
            coord_loss_value = outputs["coord_loss"].item() * trainer.accumulate_grad_batches
            report_dict["coord_loss"] = coord_loss_value

            wandb.log(report_dict, step=self.total_steps)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights")
            pl_module.save_lora(f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}")
            if pl_module.transformer.aux_token:
                torch.save(
                    pl_module.transformer.domain_embedder.state_dict(),
                    f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}/domain_embedder.pth",
                )


        # Generate and save a sample image at specified intervals
        if (self.total_steps == 1 or self.total_steps % self.sample_interval == 0) and self.test_function:
            print(f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample")
            pl_module.eval()
            self.test_function(
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
                self.total_steps,
            )
            pl_module.train()


def train(dataset, trainable_model, config, test_function):
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    # config = get_config()

    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")
    run_name = os.environ.get("WANDB_NAME", run_name)

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # Initialize dataloader
    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 1),
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # Callbacks for testing and saving checkpoints
    save_path = training_config.get("save_path", "./output")
    callbacks = [
        ModelCheckpoint(
            dirpath=f"{save_path}/{run_name}/ckpt/training",
            save_top_k=0,
            every_n_train_steps=training_config["save_interval"],
            save_last=True,
        )
    ]
    if is_main_process:
        callbacks += [
            TrainingCallback(run_name, training_config, test_function),
        ]

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
        # strategy="ddp_find_unused_parameters_true",
    )

    setattr(trainer, "training_config", training_config)
    setattr(trainable_model, "training_config", training_config)

    # Save the training config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    resume_ckpt_path = training_config.get("resume_ckpt_path", None)

    # Start training
    trainer.fit(
        trainable_model,
        train_loader,
        ckpt_path=resume_ckpt_path,
    )
