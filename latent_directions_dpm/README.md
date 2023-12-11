---
marp: true
theme: uncover
class: invert
math: mathjax

style: |
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  .reference {
    position: absolute;
    bottom: 10px;
    right: 10px;
    font-size: 18px; /* Adjust the font size as needed */
    color: #888;    /* Adjust the color as needed */
  }
  .split-image-slide {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0;
    margin: 0;
    }

---

### Interpretable Latent Directions in DPMs 

<br>
<br>
<div style="text-align: left;">

- <span style="font-size:50%"> Diffusion Models Already Have A Semantic Latent Space (ICLR 2023)
- <span style="font-size:50%"> Discovering Interpretable Directions in the Semantic Latent Space of Diffusion Models (Submitied CPVR 2023 ?)
- <span style="font-size:50%"> Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry (NIPS 2023)

---


### Papers Presented


<div style="text-align: left;">

- <span style="font-size:50%"> Diffusion Models Already Have A Semantic Latent Space 
(ICLR 2023)
- <span style="font-size:50%"> Discovering Interpretable Directions in the Semantic Latent Space of Diffusion Models 
(Submitied CVPR 2023 ?)

- <span style="font-size:50%"> Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry 
(NIPS 2023)

---

##### Diffusion Models Already Have A Semantic Latent Space 

<br>


<div style="text-align: left;">

- <span style="font-size:80%"> Main Results:

  - <span style="font-size:50%"> *h-space*: 

  - <span style="font-size:50%"> *Asymetric Reverse Process*: Small changes in UNet dont effect the inference

---


##### Diffusion Models Already Have A Semantic Latent Space 

<br>


<div style="text-align: left;">

- <span style="font-size:80%"> *h-space*:

  - <span style="font-size:50%"> $\Delta h_t$ has the same semantic effect on different samples

  - <span style="font-size:50%"> Scaling $\Delta h$ controls the magnitude of attribute chang

  - <span style="font-size:50%"> Adding multiple ∆h manipulates the corresponding multiple attribute

---




##### Diffusion Models Already Have A Semantic Latent Space 



<div style="text-align: left;">

- <span style="font-size:80%"> Why *Asymmetric Reverse Process*:

  - <span style="font-size:50%"> Let $\epsilon_\theta^t$ be a predicted noise  
  
  - <span style="font-size:50%"> Let $\hat{\epsilon}_\theta^t = \epsilon_\theta^t + \Delta \epsilon$ be its shifted counterpart. 
  - <span style="font-size:50%"> Then, $\Delta x_t = \hat{x}_{t−1} − x_t$ is negligable.
---





##### Diffusion Models Already Have A Semantic Latent Space 



<div style="text-align: left;">

- <span style="font-size:80%"> How to they perform editing?:

  - <span style="font-size:50%"> Asyrp: $x_{t-1} = \sqrt{a_{t-1}}P_t(e^\theta_t(x_t|h_t + f(h_t))) + D_t(e^\theta_T(x_t))$
  
  - <span style="font-size:50%"> Loss: $(1-\lambda)\mathcal{L}_{CLIP} + \lambda \mathcal{L}_{rec}$ 
  - <span style="font-size:50%"> They only train $f$.
---









