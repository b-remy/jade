# joint-posterior
Infering jointly cosmology and dark matter field from cosmic shear

Joint denoiser architecture using vision-transformer (or just image transformer https://arxiv.org/abs/2511.13720). DM field is decomposed into patches, then tokenised. Cosmology is also tokenised. Both set of tokens are fed to the same transformer layers. Denoised field and cosmology and predicted jointly from this architecture.

This enalbes us to have a model for the joint posteior $p(\delta_\text{DM}, \theta)$.

We can then use deep posterior sampling to sample from the joint poseterior $p(\delta_\text{DM}, \theta \mid \gamma)$, using an explicit likelihood and diffusion models.
