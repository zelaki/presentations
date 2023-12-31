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
    line-height: 0.9; /* Adjust this value as needed */
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

<<<<<<< HEAD

=======
>>>>>>> c2b237d28b58f6fef5d2a54e4bfde387c4c3e9c8
---


### Papers Presented


<div style="text-align: left;">

- <span style="font-size:50%"> **Diffusion Models Already Have A Semantic Latent Space** 
(ICLR 2023)
- <span style="font-size:50%"> **Discovering Interpretable Directions in the Semantic Latent Space of Diffusion Models** 
(Submitied CVPR 2023 ?)

- <span style="font-size:50%"> **Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry** 
(NIPS 2023)

---

##### Diffusion Models Already Have A Semantic Latent Space 

<br>


<div style="text-align: left;">

- <span style="font-size:70%"> Main Results:

  - <span style="font-size:60%"> *h-space*

  - <span style="font-size:60%"> *Asymetric Reverse Process (Asyrp)*

---


###### Diffusion Models Already Have A Semantic Latent Space 




<div style="text-align: left;">

- <span style="font-size:80%"> *h-space*:

  - <span style="font-size:50%"> $\Delta h_t$ has the same semantic effect on different samples

  - <span style="font-size:50%"> Scaling $\Delta h$ controls the magnitude of attribute change

  - <span style="font-size:50%"> Adding multiple ∆h manipulates multiple attributes

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

- <span style="font-size:80%"> How do they perform editing?:

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
  - <span style="font-size:70%"> $v_1, ..., v_m$ of $V$ yield an orthonormal basis of $T_x M$
  - <span style="font-size:70%"> $u1, ..., u_n$ of $U$ yield an orthonormal basis of $T_{\phi(x)} N$
  - <span style="font-size:70%"> $v_i$ directions in the input space that are most influential for the changes in the output space.


---

###### Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry

<div style="text-align: left;">


- <span style="font-size:70%"> Main Ideas:

  * <span style="font-size:60%"> $\mathcal{H}$ (bottleneck) exibits *local linearity* but $\mathcal{X}$ doesnt.

  * <span style="font-size:60%"> The unet encoder is a function $f: \mathcal{X} \rightarrow \mathcal{H}$

  * <span style="font-size:60%"> Let $T_x$ and $T_h$ be tangent spaces at $x\in \mathcal{X}$ and $h \in \mathcal{H}$ respectively.
  
  * <span style="font-size:60%"> The Jacobian $J_x = \nabla_x h$ is a linear map, $J_x: T_x \rightarrow T_h$

![bg right width:500px ](figs/riemannian.png)

---

###### Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry

<div style="text-align: left;">


* <span style="font-size:50%"> The right singular values $v_i$ of $J_x$ give $n$ semantic directions in $T_x$ that show
large variability of the corresponding $u \in T_h$

* <span style="font-size:50%"> The corresponding directions in $T_h$ (left singular vectors): $u_i = \frac{1}{\sigma_i} J_x v_i$

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

* <span style="font-size:70%"> Editing is performed in a single timestep.

<br>
<br>

</div>

![ width:1000px ](figs/pull_edit.png)



---


<div style="text-align: left;">



* <span style="font-size:70%">  They vectorize $x\in \mathcal{X}$ and $h \in \mathcal{H}$, thus:

  * <span style="font-size:70%">  $x \in \mathcal{X} \subset R^n$, $n=256 \cdot 256 \cdot 3$  
  * <span style="font-size:70%">  $h \in \mathcal{H} \subset R^m$, $m=8 \cdot 8 \cdot 1280$
  * <span style="font-size:70%"> Thus the Jacobian $J \in R^{m \times n}$.

 * <span style="font-size:70%"> Could we keep the tensorial structure and do HOSVD?




---

###### Discovering Interpretable Directions in the Semantic Latent Space of Diffusion Models

<div style="text-align: left;">


* <span style="font-size:70%">  Calculate the Jacobian of $e_\theta(x_t, h_t)$ with respect to $h_t$, $J_t = \frac{\partial e_t^{\theta}(x_t, h_t)}{\partial h_t}$ (everything is vectorized).


* <span style="font-size:70%"> The right singular vectors $v_i$ are directions in $\mathcal{H}$ that cause the largest change in $e_t^{\theta}$

*  <span style="font-size:70%"> Editing is performed in $\mathcal{H}$ not $\mathcal{X}$
    * <span style="font-size:70%"> $x_{t+1} = e_t^{\theta}(x_{t}, h_{t} + \alpha v^{\tau}_i)$
    * <span style="font-size:70%"> where $v^{\tau}_i$ is the $i$-th right singular vector of $J_{\tau}$ 
    
---


###### Important Notes
<div style="text-align: left;">


* <span style="font-size:70%">  $e$ and $h$ are **vectorized**.
* <span style="font-size:70%"> Directions found in timestep t are used to edit **all** timesteps.
* <span style="font-size:70%"> So for 50 timesteps and 50 singular vectors we get 2500 editing directions.



---


###### Supervised Editing
<div style="text-align: left;">


* <span style="font-size:70%">  Let a dataset $\{(x_i^+, x_i^-)\}_{i=1}^n$ of $n$ generated images with a present (+) or absent attribute (-) e.g. smile, old etc. 

* <span style="font-size:70%"> Editing direction: $v_t=\frac{1}{n} \sum_{i=1}^n(h_i^+ - h_i^-)$.




---


###### Summary
<div style="text-align: left;">


* <span style="font-size:70%"> Approach of $1^\text{st}$ paper :
  * <span style="font-size:70%"> Calculate the Jacobian of the *Unet encoder* $f: \mathcal{X} \rightarrow \mathcal{H}$, $J_t = \frac{\partial f}{\partial x_t}$.
  * <span style="font-size:70%"> The right singular vectors of $J_t$ are directions in $\mathcal{X}$ that show large variability in $\mathcal{H}$.
  * <span style="font-size:70%"> The latent variable $h_t \in \mathcal{H}$ is completely determined by $x_t \in \mathcal{X}$.
   

<br>

* <span style="font-size:70%"> Approach of $2^\text{nd}$ paper :
  * <span style="font-size:70%"> Calculate the Jacobian of the *Unet* $e_\theta: \mathcal{X} \rightarrow \mathcal{X}$ w.r.t. $h_t \in \mathcal{H}$, $J_t = \frac{\partial e_\theta}{\partial h_t}$.

  * <span style="font-size:70%"> The right singular vectors of $J_t$ are directions in $\mathcal{H}$ that show large variability in $\mathcal{X}$.

  * <span style="font-size:70%"> Since there are **skip connections**, $x_{t+1}$ depends on $h_t$ but also οn $x_t$. **$h_t$ doesn't determine the output completely**.




---


###### Our Approach
<div style="text-align: left;">


 <span style="font-size:70%"> Retain the tensorial structure of the Jacobian and perform HOSVD.
 <br>
* <span style="font-size:70%"> In the setting of the $2^{nd}$ paper:

  * <span style="font-size:70%"> $J_t = \frac{\partial e_\theta (x_t, h_t)}{\partial h_t} \in \mathbb{R}^{m \times n}$, where $m = 256 \cdot 256 \cdot 3$ and $n = 8 \cdot 8 \cdot 1280$.

  * <span style="font-size:70%">Reshape $J_t \in \mathbb{R}^{m \times H \times W \times C}$, where $H=W=8$ and $C=1280$.

  
  * <span style="font-size:70%">HOSVD on $J_t = S \times_1 U_m \times_2 U_H \times_3 U_W \times_4 U_C$.

  * <span style="font-size:70%"> Each $U_n$ is a basis vector for the space spanned by the mode-$n$ fibers.

  * <span style="font-size:70%"> Mode-wise edits: $h_t = h_t +\alpha \cdot u_i^C \circ \mathbb{1}_H \circ \mathbb{1}_W$



---
<div style="text-align: left;">

* <span style="font-size:70%"> In the setting of the $1^{st}$ paper:

  * <span style="font-size:70%"> It is not straightforward, since editing is performed in $\mathcal{X}$ not in $\mathcal{H}$.

  * <span style="font-size:70%"> If $J_t \in \mathbb{R}^{m\times n}$ SVD gives left $u_i$ and right $v_i$ singular vectors that obey $u_i = \frac{1}{\sigma_i} J_t v_i$

  * <span style="font-size:70%"> Thus directions in $\mathcal{X}$ can be mapped in $\mathcal{H}$ and vice-versa.

  * <span style="font-size:70%"> If we retain the tensorial structure of $J_t \in \mathbb{R}^{H \times W \times C \times n}$ and perform HOSVD:

    * <span style="font-size:70%"> HOSVD on $J_t = S \times_1 U_H \times_2 U_W \times_3 U_C \times_4 U_n$.

    * <span style="font-size:70%"> How can we map a basis vector $u_i^C$ to a basis vector $u_i^n$ of $\mathcal{X}$ as in SVD?

  



