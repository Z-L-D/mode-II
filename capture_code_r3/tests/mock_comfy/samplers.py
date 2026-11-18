# Minimal stand-in for comfy.samplers: CFGGuider + calc_cond_batch semantics relevant to the patch.
import torch
POOL = True          # True: cond+uncond in one batched forward; False: separate forwards
ORDER = (0, 1)       # order of branches inside a pooled batch
WRAPPER = True       # False: emulate another node dropping model_function_wrapper
TO_KWARG = False     # True: diffusion model receives transformer_options as a kwarg (real ComfyUI)
class KSAMPLER: pass
def calc_cond_batch(*a, **k): raise NotImplementedError
class CFGGuider:
    def __init__(self, model_patcher):
        self.model_patcher = model_patcher; self.cfg = 1.0; self.conds = {}
    def set_conds(self, positive, negative): self.conds = {'positive': positive, 'negative': negative}
    def set_cfg(self, cfg): self.cfg = cfg
    def predict_noise(self, x, timestep, model_options={}, seed=None):
        dm = self.model_patcher.model.diffusion_model
        ctx = {0: self.conds['positive'], 1: self.conds['negative']}
        branches = [0] if self.cfg == 1.0 else list(ORDER)
        groups = [branches] if POOL else [[b] for b in branches]
        outs = {}
        for cu in groups:
            inp = torch.cat([x] * len(cu)); c = {'ctx': torch.cat([ctx[b] for b in cu]),
                                                 'transformer_options': {'cond_or_uncond': cu[:]}}
            if TO_KWARG:
                apply_model = lambda input_x, t, **cc: dm(input_x, cc['ctx'], transformer_options=cc['transformer_options'])
            else:
                apply_model = lambda input_x, t, **cc: dm(input_x, cc['ctx'])
            if WRAPPER and 'model_function_wrapper' in model_options:
                o = model_options['model_function_wrapper'](apply_model, {'input': inp, 'timestep': timestep, 'c': c, 'cond_or_uncond': cu})
            else:
                o = apply_model(inp, timestep, **c)
            for b, oo in zip(cu, o.chunk(len(cu))): outs[b] = oo
        return outs[0] if self.cfg == 1.0 else outs[1] + self.cfg * (outs[0] - outs[1])
    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        x = noise.clone()
        mo = {'transformer_options': {'sample_sigmas': sigmas}}
        for t in sigmas[:-1]:
            x = x - 0.1 * self.predict_noise(x, t.reshape(1), mo, seed=seed)
        return x
