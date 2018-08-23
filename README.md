# Poisson Factor Analysis
Implementations of various Poisson Factor Analysis (PFA) models. 

## File Descriptions
### Models
* `BHPF.m` : Implements Gibbs sampling for [Bayesian Hierarchical Poisson Factorization (BHPF)](http://www.cs.columbia.edu/~blei/papers/GopalanHofmanBlei2015.pdf).
* `BPTF.m` : Implements Gibbs sampling for [Bayesian Poisson Tensor Factorization (BPTF)](https://people.cs.umass.edu/~aschein/ScheinPaisleyBleiWallach2015_paper.pdf).
* `BGGP.m` : 
* `GaP_v1.m` : 
* `GaP_v2.m` :
* `GaP_v3.m` :
* `HaLRTC.m` : A baseline model using low rank tensor completion ([HaLRTC](http://peterwonka.net/Publications/pdfs/2012.PAMI.JiLiu.Tensor%20Completion.pdf)).

### Tools
* `kr.m` : 
* `mat2ten.m` :
* `ten2mat.m` :
* `cp_combination.m` :
* `missing_setting.m` :    

### Running
* `main_code.m` : The main code file. Arguments `low_rank` and `missing_rate` are supposed to be modified.

### Dataset
* `tensor.mat` : See [Urban Traffic Speed Dataset of Guangzhou](https://github.com/sysuits/urban-traffic-speed-dataset-Guangzhou).
