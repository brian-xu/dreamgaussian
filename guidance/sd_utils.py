import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from utils.threestudio_utils import (
    get_text_embeddings_perp_neg,
    perpendicular_component,
)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        sd_version="2.1",
        hf_key=None,
        t_range=[0.02, 0.98],
        use_sdi=False,
    ):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device=device, dtype=self.dtype)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype
        )

        self.use_sdi = use_sdi
        if self.use_sdi:
            self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
                model_key,
                subfolder="scheduler",
                torch_dtype=self.dtype,
            )
            inversion_n_steps = 10
            self.inverse_scheduler.set_timesteps(inversion_n_steps, device=self.device)
            self.inverse_scheduler.alphas_cumprod = (
                self.inverse_scheduler.alphas_cumprod.to(device=self.device)
            )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings["pos"] = pos_embeds
        self.embeddings["neg"] = neg_embeds
        self.embeddings["pos_vd"] = []
        self.embeddings["neg_vd"] = []

        # directional embeddings
        for d in ["side", "front", "back", "overhead"]:
            pos_embeds = self.encode_text([f"{p}, {d} view" for p in prompts])
            neg_embeds = self.encode_text([f"{p}, {d} view" for p in prompts])
            self.embeddings[d] = pos_embeds
            self.embeddings["pos_vd"].append(pos_embeds)
            self.embeddings["neg_vd"].append(neg_embeds)

    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def refine(
        self,
        pred_rgb,
        guidance_scale=100,
        steps=50,
        strength=0.8,
        elevation=None,
        azimuth=None,
        camera_distances=None,
    ):

        batch_size = pred_rgb.shape[0]
        pred_rgb_512 = F.interpolate(
            pred_rgb, (512, 512), mode="bilinear", align_corners=False
        )
        latents = self.encode_imgs(pred_rgb_512.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        if self.use_sdi:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=self.dtype,
                device=self.device,
            )
            latents, noise = self.invert_noise(
                latents,
                t,
                use_perp_neg=True,
                elevation=elevation,
                azimuth=azimuth,
                camera_distances=camera_distances,
            )
        else:
            latents = self.scheduler.add_noise(
                latents, torch.randn_like(latents), self.scheduler.timesteps[init_step]
            )

        embeddings = torch.cat(
            [
                self.embeddings["pos"].expand(batch_size, -1, -1),
                self.embeddings["neg"].expand(batch_size, -1, -1),
            ]
        )

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):

            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeddings,
            ).sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]
        return imgs

    def train_step(
        self,
        pred_rgb,
        step_ratio=None,
        guidance_scale=100,
        as_latent=False,
        elevation=None,
        azimuth=None,
        camera_distances=None,
    ):

        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = (
                F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False)
                * 2
                - 1
            )
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(
                pred_rgb, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        with torch.no_grad():
            if step_ratio is not None:
                # dreamtime-like
                # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(
                    self.min_step, self.max_step
                )
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            else:
                t = torch.randint(
                    self.min_step,
                    self.max_step + 1,
                    (batch_size,),
                    dtype=torch.long,
                    device=self.device,
                )

            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

            if self.use_sdi:
                t = torch.randint(
                    self.min_step,
                    self.max_step + 1,
                    [1],
                    dtype=self.dtype,
                    device=self.device,
                )
                latents_noisy, noise = self.invert_noise(
                    latents,
                    t,
                    use_perp_neg=True,
                    elevation=elevation,
                    azimuth=azimuth,
                    camera_distances=camera_distances,
                )
            else:
                # predict the noise residual with unet, NO grad!
                # add noise
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2).to(dtype=self.dtype)
            tt = torch.cat([t] * 2).to(dtype=self.dtype)

            if azimuth is None:
                embeddings = torch.cat(
                    [
                        self.embeddings["pos"].expand(batch_size, -1, -1),
                        self.embeddings["neg"].expand(batch_size, -1, -1),
                    ]
                )
            else:

                def _get_dir_ind(h):
                    if abs(h) < 60:
                        return "front"
                    elif abs(h) < 120:
                        return "side"
                    else:
                        return "back"

                embeddings = torch.cat(
                    [self.embeddings[_get_dir_ind(h)] for h in azimuth]
                    + [self.embeddings["neg"].expand(batch_size, -1, -1)]
                ).to(dtype=self.dtype)

            noise_pred = self.unet(
                latent_model_input, tt, encoder_hidden_states=embeddings
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            # seems important to avoid NaN...
            # grad = grad.clamp(-1, 1)

        target = (latents - grad).detach()
        loss = (
            0.5
            * F.mse_loss(latents.float(), target, reduction="sum")
            / latents.shape[0]
        )

        return loss

    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    1,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        batch_size = latents.shape[0]
        self.scheduler.set_timesteps(num_inference_steps)
        embeddings = torch.cat(
            [
                self.embeddings["pos"].expand(batch_size, -1, -1),
                self.embeddings["neg"].expand(batch_size, -1, -1),
            ]
        )

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings
            ).sample

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)

        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs

    # From SDI
    @torch.no_grad()
    def invert_noise(
        self,
        start_latents,
        invert_to_t,
        use_perp_neg,
        elevation,
        azimuth,
        camera_distances,
    ):
        latents = start_latents.clone()
        B = start_latents.shape[0]

        timesteps = self.get_inversion_timesteps(invert_to_t, B)
        for t, next_t in zip(timesteps[:-1], timesteps[1:]):
            noise_pred, _, _ = self.predict_noise(
                latents,
                t.repeat([B]),
                use_perp_neg,
                elevation,
                azimuth,
                camera_distances,
                guidance_scale=-7.5,
            )
            latents = self.ddim_inversion_step(noise_pred, t, next_t, latents)

        # remap the noise from t+delta_t to t
        found_noise = self.get_noise_from_target(start_latents, latents, next_t)

        return latents, found_noise

    @torch.no_grad()
    def predict_noise(
        self,
        latents_noisy,
        t,
        use_perp_neg,
        elevation,
        azimuth,
        camera_distances,
        guidance_scale: float = 1.0,
        text_embeddings=None,
    ):

        batch_size = elevation.shape[0]

        if use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = get_text_embeddings_perp_neg(
                self.embeddings["pos_vd"],
                self.embeddings["neg_vd"],
                elevation,
                azimuth,
                camera_distances,
                True,
            )

            latent_model_input = torch.cat([latents_noisy] * 4, dim=0).to(self.dtype)
            text_embeddings = text_embeddings.squeeze().to(self.dtype)
            noise_pred = self.unet(
                latent_model_input,
                torch.cat([t] * 4),
                encoder_hidden_states=text_embeddings,
            ).sample  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(-1, 1, 1, 1).to(
                    e_i_neg.device
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + guidance_scale * (e_pos + accum_grad)
        else:
            neg_guidance_weights = None

            if text_embeddings is None:
                text_embeddings = self.embeddings["pos"]
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred, neg_guidance_weights, text_embeddings

    def ddim_inversion_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.FloatTensor,
        inversion_eta=0.3,
    ) -> torch.FloatTensor:
        # 1. compute alphas, betas
        # change original implementation to exactly match noise levels for analogous forward process
        alpha_prod_t = (
            self.inverse_scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.inverse_scheduler.initial_alpha_cumprod
        )
        alpha_prod_t_prev = self.inverse_scheduler.alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.inverse_scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.inverse_scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
        elif self.inverse_scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (
                beta_prod_t**0.5
            ) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.inverse_scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )
        # 3. Clip or threshold "predicted x_0"
        if self.inverse_scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.inverse_scheduler.config.clip_sample_range,
                self.inverse_scheduler.config.clip_sample_range,
            )
        # 4. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # 5. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )

        # 6. Add noise to the sample
        variance = self.scheduler._get_variance(prev_timestep, timestep) ** (0.5)
        prev_sample += inversion_eta * torch.randn_like(prev_sample) * variance

        return prev_sample

    def get_inversion_timesteps(self, invert_to_t, B, inversion_n_steps=10):
        n_training_steps = self.inverse_scheduler.config.num_train_timesteps
        effective_n_inversion_steps = inversion_n_steps  # int((n_training_steps / invert_to_t) * self.cfg.inversion_n_steps)

        if self.inverse_scheduler.config.timestep_spacing == "leading":
            step_ratio = n_training_steps // effective_n_inversion_steps
            timesteps = (
                (np.arange(0, effective_n_inversion_steps) * step_ratio)
                .round()
                .copy()
                .astype(np.int64)
            )
            timesteps += self.inverse_scheduler.config.steps_offset
        elif self.inverse_scheduler.config.timestep_spacing == "trailing":
            step_ratio = n_training_steps / effective_n_inversion_steps
            timesteps = np.round(
                np.arange(n_training_steps, 0, -step_ratio)[::-1]
            ).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.inverse_scheduler.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )
        # use only timesteps before invert_to_t
        timesteps = timesteps[timesteps < int(invert_to_t)]

        # Roll timesteps array by one to reflect reversed origin and destination semantics for each step
        timesteps = np.concatenate([[int(timesteps[0] - step_ratio)], timesteps])
        timesteps = torch.from_numpy(timesteps).to(self.device)

        # Add the last step
        delta_t = int(
            random.random()
            * self.inverse_scheduler.config.num_train_timesteps
            // inversion_n_steps
        )
        last_t = torch.tensor(
            min(  # timesteps[-1] + self.inverse_scheduler.config.num_train_timesteps // self.inverse_scheduler.num_inference_steps,
                invert_to_t + delta_t,
                self.inverse_scheduler.config.num_train_timesteps - 1,
            ),
            device=self.device,
        )
        timesteps = torch.cat([timesteps, last_t.repeat([B])])
        return timesteps

    def get_noise_from_target(self, target, cur_xt, t):
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        noise = (cur_xt - target * alpha_prod_t ** (0.5)) / (beta_prod_t ** (0.5))
        return noise


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1",
        choices=["1.5", "2.0", "2.1"],
        help="stable diffusion version",
    )
    parser.add_argument(
        "--hf_key",
        type=str,
        default=None,
        help="hugging face Stable diffusion model key",
    )
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
