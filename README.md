### Introduction

Large Backward Reach Avoid Tubes

### Examples

+ [Basic Double Integrator](/Examples/dint_basic.py): Time to reach the origin for the double integrator with switching curve dynamics.

+ [Robustly Controlled Backward Reachable Tube -- Two Dubins Vehicles in Relative Coordinates](/Examples/dubins_rel.py): Evaluate the backward reachable tube for two Dubins vehicles in relative coordinates. The pursuer is at the origin.

+ [Murmurations of Swarms -- Collective behavior from Topological interactions](/Examples/murmurations.py)

### Evolution of BRAT for Different Flocks

Here, we initialized six flocks on a state space to constitute a simple murmuration's trajectory verification. Within each flock are 
6 or 7 individual agents, whose trajectories must respect certain safety constraints.  We evolve the trajectories over a time horizon \\(-100secs \le t \le 0\\) so that at the end of each integrattion runs, we can obtain the _robustly controllable backward reach-avoid tubes_ for each flock within the system.

<div align="center">
<img src="BRATVisualization/flock_00.gif" height="360px" width="340px"/>
<img src="BRATVisualization/flock_01.gif" height="360px" width="340px"/>
</div>


<div align="center">
<img src="BRATVisualization/flock_02.gif" height="360px" width="340px"/>
<img src="BRATVisualization/flock_03.gif" height="360px" width="340px"/>
</div>


<div align="center">
<img src="BRATVisualization/flock_04.gif" height="360px" width="340px"/>
<img src="BRATVisualization/flock_05.gif" height="360px" width="340px"/>
</div>

<div align="center">
<img src="BRATVisualization/flock_06.gif" height="640px" width="480px"/>
</div>

RCBRAT because each agent within each flock must avoid other agents, while as a group agents within the flock are trying to evade capture by a pursuer.