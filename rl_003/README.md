# REINFORCE for BipedalWalker-v3

## アルゴリズム

## 算数

### log_prod

$$
\pi_\theta(a|s) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(a-\mu)^2}{2\sigma^2}\right)
$$

両辺に$\log$をかけると

$$\begin{align}
\log(\pi_\theta(a|s)) &= \log\left(\frac{1}{\sqrt{2\pi\sigma^2}} \cdot \exp\left(-\frac{(a-\mu)^2}{2\sigma^2}\right)\right) \notag\\

&= -\log(\sqrt{2\pi\sigma^2}) - \frac{(a-\mu)^2}{2\sigma^2} \notag\\

&= -\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2}\frac{(a-\mu)^2}{\sigma^2} \notag\\

&= -\frac{1}{2}\left(\log(2\pi\sigma^2) + \left(\frac{a-\mu}{\sigma}\right)^2\right) \notag\\

&= -\frac{1}{2}\left(\log(2\pi) + 2\log\sigma + \left(\frac{a-\mu}{\sigma}\right)^2\right) \notag\\

&= -\frac{1}{2}\left(\log(2\pi) + \left(\frac{a-\mu}{\sigma}\right)^2\right) - \log\sigma \notag\\
\end{align}$$

