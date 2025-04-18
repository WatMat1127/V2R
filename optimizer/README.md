This is the latest version of the optimizer for the VLAO calculation (at April 9, 2025). For the original version, please see [VLAO/optimizer](https://github.com/WatMat1127/VLAO/tree/main/optimizer).

# How to Run
## Step 0. Set GRRMsub command
Set up your computer enviroment so that a GRRM job can be executed with the following command: `GRRMsub filename`.  
An example csh script for GRRMsub is shown below:  
    
```sh
#!/bin/csh

set fn_run = "$1"
setenv GRRMroot /home/local/GRRM23
setenv PATH "${GRRMroot}:${PATH}"
setenv subgrr GRRM.out
setenv subgau g16
setenv subchk formchk
setenv subuchk unfchk
setenv GAUSS_SCRDIR /scr/

GRRMp $fn_run -p1 -s172800
```

The variable `GRRMroot` sets the location where GRRM23p and GRRM23.out are installed in your computer. For further instructions on how to run GRRM jobs, see [AFIR-web](https://afir.sci.hokudai.ac.jp).

## Step 1. Prepare input files
The following files are required. You can find elample files in the `Example` directrory.  
- _input_.txt  
- _GRRM_job_i_.com ($i = 1, \dotsc, n$)
- _GRRM_job_i_.param ($i = 1, \dotsc, n$)


### _input_.txt
An input file for `main.py` which specifies GRRM jobs to be executed, the absolute path for `VL_main.py`, parameters to be optimized, and other detailed settings for optimization in the following format:

  ----------
  \# GRRMjobs to be executed    
  com_1: _GRRM_job_1_, (float)   
  com_2: _GRRM_job_2_, (float)   
  •••
  
  \# the absolute path for VL_main.py  
  SubAddExPot=_xxx_/_yyy_/VL_main.py  
  
  \# parameter setting  
  param_tag  : $p_1^\mathrm{label}$, $p_2^\mathrm{label}$, $p_3^\mathrm{label}$, ..., $p_N^\mathrm{label}$  
  init_param : $p_1^\mathrm{init}$, $p_2^\mathrm{init}$, $p_3^\mathrm{init}$, ..., $p_N^\mathrm{init}$  
  param_range: $p_1^\mathrm{low}$ _ $p_1^\mathrm{high}$, $p_2^\mathrm{low}$ _ $p_2^\mathrm{high}$, $p_3^\mathrm{low}$ _ $p_3^\mathrm{high}$ ..., $p_N^\mathrm{low}$ _ $p_N^\mathrm{high}$  
  penalty_std: $P_0$
  
  \# detailed setting  
  max_itr         : (int)   
  grad_threshold  : (float)   
  param_threshold : (float)   
  f_val_threshold : (float)   

The `com_n` statement specifies GRRMjob to be executed (format: com\__n_: _filename_ (no extension), _energy correction_).  
The `SubAddExPot` statement specifies the absolute path for `VL_main.py`.    
The `param_tag` statement specifies the parameters to be optimized. The label should be consistent with those written in _GRRM_job_i_.param.      
The `init_param` statement specifies the initial values for each paramter.    
The `param_range` statement specifies the lower and higher bounds for each paramter (see the function B(p) in the original paper).  
The `penalty_std` statement specifies the scale of the barrier function (see the function B(p) in the original paper).  
The `max_itr` statement specifies the maximum number of iteration for the conjugate gradient method.   
The `grad_threshold` and `param_threshold` statements specify termination criteria. The calculation will be terminated when the maximum absolute values among the gradient components and the displacement components are smaller than those specified here.  
The `f_val_threshold` statement specifies another termination criterion. The calculation will be terminated when the objective function is smaller than those specified here.  
   

### _GRRM_job_i_.com
An input file for the GRRM23 program. See [AFIR-web](https://afir.sci.hokudai.ac.jp) for the detailed format. The `@@SubAddExPot@@` needs to be specified instead of `SubAddExPot=xxx/yyy/VL_main.py` in the option part.

### _GRRM_job_i_.param
An input file for the VL program. See the `README` in the `VL` directory for detail. The parameter to be optimized must be specified as "@@_p<sub>j</sub>_<sup>label</sup>@@" instead of corresponding initial values. The label must be consistent with those written in _input_.txt.  

## Step 2. Modify z_function.py
An arbitrary objective function can be implemented as the `calc_f_val_grad` function. The `f_val` specified here will be minimized. 
You can find example files in this directrory. The format for the `calc_f_val_grad` function is as follows:

```python
def calc_f_val_grad(qm_ene_list, grad_list):
    •••
    (calculation of f_val and f_grad)
    •••
    return f_val, f_grad
```

The `f_val` and `f_grad` are an objective function and a list of its gradients with respect to the VL parameters to be optimized.  
The `qm_ene_list` argument is a list containing the electronic energy of _GRRM_job_i_ (corrected by the corresponding _energy correction_ specified in  _input_.txt) as the *i*th component.  
The `grad_list` argument is a list of `numpy.ndarray`, where the *j*th component of the *i*th `ndarray` is corresponds to a derivative value of the *i*th electronic energy with respect to the *j*th parameter.

## Step 3. Run calculation
The calculation can be executed with the following command: `python main.py input run`.  
As the optimization usually takes some time, it is recommended to run the calculation background with the following command:  
`nohup python main.py input run >& result.log &`.  

Immediately, _GRRM_job_i_\_step0.com and _GRRM_job_i_\_step0.param ($i = 1, \dotsc, n$) will be generated and corresponding GRRM jobs will be executed. After several steps, a file named analysis.txt will appear, summarizing the results of the parameter optimization.  
The `f_val_log` shows the objective function value at each iteration. The values with and without the barrier function will be reported.  
The `f_grad_log` shows the derivative values of the objective function with respect to each VL parameter. These values will be reported in the order specified by `param_tag` in _input_.txt.   
The `param_log` shows VL parameters at each iteration. These values will be reported in the order specified by `param_tag` in _input_.txt.

