"""Reuse frozen review renderer with parameter counts taken from evaluation."""
from pathlib import Path
original=Path(__file__).with_name('render_joint_review.py')
source=original.read_text(encoding='utf-8')
old='Joint: 40 latent + 6 pose variables'
new='Joint: {evaluation["active_latent_dimensions"]} latent + {evaluation["total_dimensions"]-evaluation["active_latent_dimensions"]} pose variables'
if source.count(old)!=1: raise ValueError('legacy renderer caption changed')
source=source.replace(old,new)
exec(compile(source,str(original),'exec'),{'__name__':'__main__','__file__':str(original)})
