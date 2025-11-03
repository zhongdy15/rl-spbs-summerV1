Additional requirements for running the SemiPhysBuildingSim environment:
pythermalcomfort == 2.10.0


The variable  USE_Multi_Discrete(in SemiPhysBuildingSim/SemiPhysBuildingSim.py) means the type of action space.

[USE_Multi_Discrete = True]
python bdq_test.py 


[USE_Multi_Discrete = False]
python dqn_test.py 