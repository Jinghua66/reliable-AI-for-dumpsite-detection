"""Generate SD v1.5 + LoRA patches. Requires a CUDA environment."""
import argparse
import json
from pathlib import Path
import numpy as np
from .augment import load_manifest
from .voc import sha256

PROMPTS = {
    'agriculture forestry': 'agricultural waste yard:1.5, crop waste,organics, remote sensing image',
    'construction waste': 'construction yard:1.5, demolition waste:1.5, concrete, remote sensing image, distinguishable',
    'domestic garbage': 'domestic garbage dumpsite:1.5, municipal solid waste:1.5, remote sensing image, complicated component',
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model', required=True, help='Exact SD v1.5 local directory or repository ID')
    p.add_argument('--revision', help='Required immutable commit when using a remote repository')
    p.add_argument('--lora', default='weights/lora-15000.safetensors')
    p.add_argument('--manifest', default='examples/manifest.csv')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if not Path(a.base_model).is_dir() and not a.revision:
        p.error('Remote base model requires --revision')
    import torch
    import diffusers
    from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    pipe = AutoPipelineForText2Image.from_pretrained(a.base_model, revision=a.revision,
        torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
        low_cpu_mem_usage=False, use_safetensors=True).to(a.device)
    lora = Path(a.lora)
    pipe.load_lora_weights(str(lora.parent), weight_name=lora.name)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    trace = []
    seen = set()
    for row in load_manifest(a.manifest):
        filename = row['patch_file']
        if Path(filename).name != filename:
            raise ValueError('Patch filename must not contain directories')
        if filename in seen: continue
        seen.add(filename)
        seed = int(row['generation_seed'])
        guidance = np.random.RandomState(seed).randint(50, 150) / 10
        if row['class'] == 'construction waste': guidance /= 1.5
        # Construction uses a lower guidance scale; the other classes use the sampled scale.
        image = pipe(PROMPTS[row['class']], guidance_scale=guidance,
            generator=torch.Generator(device=a.device).manual_seed(seed),
            num_inference_steps=25, eta=0.2).images[0]
        image.save(out / filename)
        trace.append(dict(row, prompt=PROMPTS[row['class']], guidance_scale=guidance,
                          output_sha256=sha256(out / filename)))
    (out / 'generation.json').write_text(json.dumps(dict(base_model=a.base_model,
        revision=a.revision, lora_sha256=sha256(lora), steps=25,
        torch=torch.__version__, diffusers=diffusers.__version__, outputs=trace), indent=2))

if __name__ == '__main__': main()
