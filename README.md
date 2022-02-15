## ComplexBRAT: Complex Backward Reach Avoid Tubes

 Code for the Hamilton-Jacobi-Isaacs Analysis of the murmurations paper: [ComplexBRAT: Complex Backward Reach Avoid Tubes](Examples/murmurations.py).

For the technical details on the theory behind this work, please see this paper: 

```
@article{ComplexBRAT,
title   = {ComplexBRAT: Complex Backward Reach-Avoid Tubes. An Emergent Collective Behavior Framework.},
author  = {Ogunmolu, Olalekan.},
journal = {Algorithm Foundations of Robotics, XV (WAFR).},
year    = {2022},
}
```

A preprint can be downloaded here: [ComplexBRAT: Complex Backward Reach-Avoid Tubes. An Emergent Collective Behavior Framework.](Papers/wafr.pdf)

#### Evolution of BRAT for Different Flocks

Here, we initialized various flocks on a state space to constitute a simple murmuration's trajectory verification. Within each flock are 
6 or 7 individual agents, whose trajectories must respect certain safety constraints.  We evolve the trajectories over a time horizon _-100 <= t <= 0_ so that at the end of each integration run, we obtain the _robustly controllable backward reach-avoid tube_ or (**RCBRAT**) for each flock within the system.

<div align="center">
<img src="BRATVisualization/flock_01.gif" height="330px" width="260px"/>
<img src="BRATVisualization/flock_02.gif" height="330px" width="260px"/>
<img src="BRATVisualization/flock_03.gif" height="330px" width="260px"/>
</div>


<div align="center">
    <img src="BRATVisualization/flock_04.gif" height="330px" width="260px"/>
    <img src="BRATVisualization/flock_05.gif" height="330px" width="260px"/>
    <img src="BRATVisualization/flock_06.gif" height="330px" width="260px"/>
</div>

RCBRAT because each agent within each flock must avoid other agents that fall within a circle constructed from a pre-specified radius defined on its body frame; while as a group/murmuration, all agents must evade capture by a (multiple) pursuing attacker (s).

#### Fast Stitching of 𝜖-BRATs via Voronoi Implicit Interface Methods

The stiching together of the respective BRATs are then carried out using a variation of the **Voronoi Implicit Interface Method** and the so-called "_redistancing_" or "_reinitialization_" method in level set methods.

### Setup

It's best to create a virtual or conda environment in python 3.6+ (I used Python 3.8/3.9) to reproduce the results in this paper.

#### **Dependencies** 

+ All the dependencies listed below are installable via: `pip install -r requirements.txt`

| Dependency      | Dependency      | Dependency      | 
| :--:     | :---:               | :---:               | 
| [H5py](https://www.h5py.org/) | [Numpy](https://numpy.org/)  | [Cupy](https://cupy.dev/) |
| [LevelSetPy](https://github.com/robotsorcerer/LevelSetPy)   | [Scipy](https://scipy.org/) | [Scikit-image](https://scikit-image.org/)     | 

### Running

```
    python Examples/murmurations.py <options>
```

**Options**:
* `--flock_num`: Label of a flock within a murmuration to run. This is useful for single agent optimization Defaults to an `int` type.
* `--save`: Save the BRAT at the end of each integration step? Defaults to a `bool` type.
* `--out_dir`: Directory in which to dump the BRATs. Defaults to `'./data'`.
* `--visualize`: Visualize the flock's zero-level set? Defaults to a `bool` type.
* `--flock_payoff`: Should we compute the payoff of (a/every) flock? Defaults to a `bool` type.
* `--resume`: Should we resume the optimization from a previously computed BRAT iteration? Defaults to a `str` type.
* `--mode`: What mode are we running the computation in? For individual flocks or stitching together the BRATs of computed flocks? Defaults to a `str` type.
* `--verify`: Whether we want to verify a murmuration's trajectory after the optimization? Defaults to a `bool` type.
* `--elevation`: What elevation angle should we display the BRAT if `visualize` is set to `True`? Defaults to `float` type.
* `--azimuth`: What azimuth angle should we display the BRAT if `visualize` is set to `True`? Defaults to `float` type.
* `--pause_time`: How many seconds to wait between displaying a zero-level set on pyplot? Defaults to `float` type.


### Further Examples 

Note that these do not have anything to do with murmurations or emergent collective behavior.

+ [Basic Double Integrator](Examples/dint_basic.py): Time to reach the origin for the double integrator with switching curve dynamics.
+ [Robustly Controlled Backward Reachable Tube -- Two Dubins Vehicles in Relative Coordinates](Examples/dubins_rel.py): Evaluate the backward reachable tube for two Dubins vehicles in relative coordinates. The pursuer is at the origin.