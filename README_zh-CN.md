$$
\begin{array}{l}
P(z_{n} \geq z_{n^{\prime}} ; \forall n^{\prime} \neq n \mid\{\pi_{n^{\prime}}\}_{n^{\prime}=1}^{N})\\
=\int \prod_{n^{\prime} \neq n} e^{-e^{-(z_{n}-\pi_{n^{\prime}})}} \cdot e^{-(z_{n}-\pi_{n})-e^{-(z_{n}-\pi_{n})}} d z_{n}\\
=\int e^{-\sum_{n^{\prime} \neq n} e^{-(z_{n}-\pi_{n})}-(z_{n}-\pi_{n})-e^{-(z_{n}-\pi_{n})}} d z_{n}\\
=\int e^{-\sum_{n=1}^{N} e^{-(z_{n}-\pi_{n^{\prime}})}-(z_{n}-\pi_{n})} d z_{n}\\
=\int e^{-(\sum_{n=1}^{N} e^{\pi_{n^{\prime}}}) e^{-z_{n}}-z_{n}+\pi_{n}} d z_{n}\\
=\int e^{-e^{-z_{n}+\ln (\sum_{n=1}^{N}} e^{\pi^{\pi} n})_{-z_{n}+\pi_{n}} d z_{n}}\\
=\int e^{-e^{-(z_{n}-\ln (\sum_{n=1}^{N}} e^{\pi_{n^{\prime}}}))}(z_{n}-\ln (\sum_{n^{\prime}=1}^{N} e^{\pi_{n^{\prime}}}))-\ln (\sum_{n^{\prime}=1}^{N} e^{\pi^{\prime}} n^{\prime})+\pi_{n} d z_{n}\\
=e^{-\ln (\sum_{n^{\prime}}^{N} e^{e} e^{\pi_{\prime}})+\pi_{n}} \int e^{-e^{-(z_{n}-\ln (\sum_{n}^{N}=1} e^{\pi_{n^{\prime}}}))}(z_{n}-\ln (\sum_{n^{\prime}=1}^{N} e^{\pi_{n^{\prime}})} d z_{n}\\
=\frac{e^{\pi_{n}}}{\sum_{n^{\prime}=1}^{N} e^{\pi_{n^{\prime}}}} \int e^{-e^{-(z_{n}-\ln (\sum_{n}^{N}=1} e^{\pi^{\prime}}{ }_{n}^{\prime}))}(z_{n}-\ln (\sum_{n^{\prime}=1}^{N} e^{.\pi_{n^{\prime}})}) d z_{n}\\
=\frac{e^{\pi_{n}}}{\sum_{n=1}^{N} e^{\pi_{n^{\prime}}}} \int e^{-(z_{n}-\ln (\sum_{n=1}^{N} e^{\pi_{n^{\prime}}}))-e^{-(z_{n}-\ln (\sum_{n}^{N}=1} e^{\pi_{n^{\prime}}})} d z_{n}
\end{array}
$$