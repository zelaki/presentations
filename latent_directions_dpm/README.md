---
marp: true
theme: uncover
# class: invert
math: mathjax

style: |
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  section {
    line-height: 0.8; /* Adjust this value as needed */
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


###### Diffusion Models Already Have A Semantic Latent Space 




<div style="text-align: left;">

- <span style="font-size:80%"> *h-space*:

  - <span style="font-size:50%"> $\Delta h_t$ has the same semantic effect on different samples

  - <span style="font-size:50%"> Scaling $\Delta h$ controls the magnitude of attribute chang

  - <span style="font-size:50%"> Adding multiple ∆h manipulates the corresponding multiple attribute

</div>

![bg right width:600px ](figs/unet.png)


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









##### Preliminaries


<div style="text-align: left;">


- <span style="font-size:70%"> Push-Forward
<!-- - <span style="font-size:70%"> Metric Pullback -->
- <span style="font-size:70%"> SVD of a Jacobian

![bg right width:600px ](figs/Pushforward.png)


---

##### Push-Forward

- <span style="font-size:70%"> A linear approximation of smooth maps on tangent spaces.
- <span style="font-size:70%">  Let, $\phi: M \rightarrow N$
- <span style="font-size:70%"> For $x \in M$, the *Jacobian* of $\phi$ at $x$ is a linear map between their tangent spaces:
  $J_\phi:T_x M \rightarrow T_{\phi(x)} N$


![bg right width:500px ](figs/Pushforward.png)

---

##### SVD of the Jacobian


<div style="text-align: left;">

- <span style="font-size:70%"> Let $J_\phi: T_x M \rightarrow T_{\phi(x)} N$.

- <span style="font-size:70%"> Then $J_\phi = U \Sigma V^*$:
  - <span style="font-size:70%"> $u_1, ..., u_m$ of $U$ yield an orthonormal basis of $T_x M$
  - <span style="font-size:70%"> $v1, ..., v_n$ of $V$ yield an orthonormal basis of $T_{\phi(x)} N$
  - <span style="font-size:70%"> $v_i$ directions in the input space that are most influential for the changes in the output space.


---

###### Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry

<div style="text-align: left;">


- <span style="font-size:70%"> Main Ideas:

  * <span style="font-size:70%"> $\mathcal{H}$ (bottleneck) exibits local linearity but $\mathcal{X}$ doesnt.

  * <span style="font-size:70%"> The unet encoder is a function $f: \mathcal{X} \rightarrow \mathcal{H}$

  * <span style="font-size:70%"> Let $T_x$ and $T_h$ be tangat spaces at $x\in \mathcal{X}$ and $h \in \mathcal{H}$ respectivly.
  
  * <span style="font-size:70%"> The Jacobian $J_x = \nabla_x h$ is a linear map, $J_x: T_x \rightarrow T_h$

![bg right width:500px ](figs/riemannian.png)

---

###### Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry

<div style="text-align: left;">


* <span style="font-size:50%"> The right singular values $v_i$ of $J_x$ give $n$ semantic directions in $T_x$ that show
large variability of the corresponding $u \in T_h$

* <span style="font-size:50%"> The corresponding directions in $T_h$ (left singular values): $u_i = \frac{1}{\sigma_i} J_x v_i$

![bg right width:500px ](figs/riemannian.png)



---

##### Editing

<div style="text-align: left;">

<span style="font-size:50%"> 1. Map a sample in X into a
tangent space $T_h$ in H.

<span style="font-size:50%"> 2. Choose a direction in $T_h$.

<span style="font-size:50%"> 3. Find its corresponding direction in X using $J_x^{-1}$.

<span style="font-size:50%"> 4. Edit the sample by adding the discovered direction.

<span style="font-size:50%"> 5. Map the edited sample to a new tangent space $T_h'$ in $\mathcal{H}$ for multiple editing.

<span style="font-size:50%"> 6. Project to new tangent and goto 3.



![bg right width:500px ](figs/riemannian.png)



--- 


##### In a new version of the paper 

<div style="text-align: left;">

<span style="font-size:50%"> 1. Map a sample in X into a
tangent space $T_h$ in H.

<span style="font-size:50%"> 2. Choose a direction $u_i \in T_h$.

<span style="font-size:50%"> 3. Find its corresponding direction $v_i \in T_x$ using $J_x^{-1}$.

<span style="font-size:50%"> **4. $\mathbb{\hat{x_t} = x_t + \gamma [e_\theta(x_t + v_i) - e_\theta(x_t)]}$**

![bg right width:500px ](figs/riemannian.png)




---



##### Important Notes

<div style="text-align: left;">

* <span style="font-size:70%"> The editing is performed in a single timestep.

<br>
<br>

</div>

![ width:1000px ](figs/pull_edit.png)



---

##### Important Notes

<div style="text-align: left;">



* <span style="font-size:70%">  They vectorize $x\in \mathcal{X}$ and $h \in \mathcal{H}$, thus:

  * <span style="font-size:70%">  $x \in \mathcal{X} \subset R^n$, $n=256 \cdot 256 \cdot 3$  
  * <span style="font-size:70%">  $h \in \mathcal{H} \subset R^m$, $m=8 \cdot 8 \cdot 1280$
  * <span style="font-size:70%"> Thus the Jacobian $J \in R^{m \times n}$.

 * <span style="font-size:70%"> Could we keep the tensorial structure and do HOSVD?




---

###### Discovering Interpretable Directions in the Semantic Latent Space of Diffusion Models

<div style="text-align: left;">


* <span style="font-size:70%">  They take the Jacobian $e_\theta$ with respect to $h$, $J = \frac{\partial e_t^{\theta}(x_t, h_t)}{\partial h_t}$ (everything is vectorized).


* <span style="font-size:70%"> The right singulare vectors are directions in $\mathcal{H}$ that cause the largest change in $e_t^{\theta}$

*  <span style="font-size:70%">  


<!-- ![bg right width:500px ](figs/riemannian.png) -->



