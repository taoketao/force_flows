# force_flows
To find functions that lie 'partway' between f(x)=x and g(x)=e^x, this project simulates flows according to approximations of force diagrams inspired by physics dynamical systems.

Context, motivation, and the Rho problem. Coming soon: a proper writeup about the following concepts. To model neuromodulatory neurotransmitters in a neural network and their ability to generalize skills, simple sum units (*y_i* = *w_i^T* [dot] *x*) do not suffice, because they suffer from the 'echo' problem where any one neuron has negligible influence on outputs. Something akin to multiplication (*y_i* = *w_i^T* \[dot\] (*x^(1)* [elementwise-mult] *x^(2)*) is more appropriate. Modern machine learning models have been implicitly implementing functional units with multiplications in them; for a survey on so-called higher-order networks, see here[https://taoketao.github.io/docs/Neuroscience_Final_Paper.pdf]). However, multiplication units also suffer from a complimentary 'veto' problem, where a single neuron outputting a value of zero annihilates the information conveyed by all the rest of the neurons. Neither of these models are biologically plausible, either -- any neuromodulator has significant but not total control over the receptors they modulate. To resolve these fundamental issues, I've been seeking a function 'between' addition and multiplication. This is in terms of both big-O complexity and also constituents making up intermediate complexities, such as the multiplication of N and log(N) in O(N\*logN). Such a solution I call 'Rho', which is the Greek letter between Sigma (addition) and Pi (multiplication). Once a suitable function is found, there are numerous ways to restructure a neural network. For several reasons including x\*y = exp{1}(log{1}(x)+log{1}(y) and x+y=exp{0}(log{0}(x)+log{0}(y), where f{N}(x) refers to N iterates of function f(x) such that f{3}(x)=f(f(f(x))), the problem of finding a function between addition and multiplication reduces to finding a half-exponential function, exp{1/2}. Such a problem has been approached by Godfrey and Gasler 2015 [https://arxiv.org/abs/1602.01321] and by Urban and Smagt 2016 [https://arxiv.org/abs/1503.05724], but both report methods that are shockingly unsatisfying, in that their interpolation functions (a) rely on parameters that the network itself learns thus admitting the 'veto' problem (coming soon: rigorous explanation) and (b) are not monotonic interpolations of the fractional iterates of the exponential function (see their plots of 3~7 and 2~7 respectively where ~ marks their interpolation). The general problem of finding a half-exponential function, or compositional square root of exp(x), is an interesting one, because it's not readily available: for example, finding a satisfying exp{\beta} for \beta>1 or \beta<0 is easier than finding an exp{\alpha}, 0<\alpha<1; it's been shown that such a half-exponential function would imply find a cardinality that is between the countable infinity |N| and the uncountable infinity |R|, which would resolve the continuum hypothesis, a problem that is certainly undecidable in ZFC theory; finding a functional root of Lambert's W function; and numerous other examples which are listed here[https://taoketao.github.io/rho-thoughts.html]). However, a child can see that you can draw a line between the Identity and Exponential functions. And from an information theoretic standpoint and the theory of homotopies, such functions ought to exist. (Ex: [https://web.archive.org/web/20210716130446/https://en.wikipedia.org/wiki/Half-exponential_function]). However, it's clear that neuromodulators systematically do calculations between addition and multiplication; while this may reduce to numerical changes in situ, I'm interested in finding a function that isn't limitied by numerical changes, which instead qualitatively performs operations in a Rho regime, both as a possible model of neural activity as well as a new kind of deep neural network.

In pursuit of finding half-exponential functions, I've found a new promising model inspired by physics and that applies to dynamical systems. (Note, this is different from the majority of other similarities found between neural networks and dynamical systems involving grid cells, recurrent networks, attractors, and spin glasses; work of mine in this direction can be found here[https://github.com/taoketao/random_matrices_and_eigenvalues].) This model forgoes algebraic manipulations, any appeal to information theoretic constraints on neuron information, or tricks involving dimension manipulation or reliance on data distributions. Instead, it considers the following scenario:

The model. Consider the lines f(x)=x and g(x)=exp(x) to have positive electromagnetic charges spread uniformly on the curve (ie, charge per interval (x+\epsilon, x-\epsilon) is equal for all x and \epsilon). (Coming soon: charge distributed evenly on the curve itself.) Look at all the points between these two curves as having a negative charge. Then consider physics actions on these points over time. Are there any fixed points that are motionless? Is there a curve for which any point on the curve stays on it over time, and if so, do the point charges tend to flow in one direction or another? Which force functions converge and which diverge? These questions are best solved by integrals, but since some of the integrals are nefarious (ie, integral of dx/(1+(e^x-x)^2) is not readily solvable to my knowledge), I'm running some simulations that approximate the integral with discrete sums to get an intuition to the possibilities. 

Main problem. given f(x) and g(x), for the points between *z*=\[u,v\] (ie, where for a given u, u\<v\<e^u), calculate the force on a given point according to the following physics-inspired force calculation.
- Pick a distance metric, such as d(*a*,*b*) = 1/(sqrt((a1-b1)^2 + (a2-b2)^2))
- For each point along *w*=\[x,f(x)\] and along \[x,g(x)\] (ie, [x,x] and [x,e^x]), calculate the force on *z* by by *w* via formula (*w*-*z*) * d(*w*,*z*) / L2(*w*,*z), where L2 is the l-2 norm used to normalize vectors to unit length.
- Accumulate all the forces from across each *w*
