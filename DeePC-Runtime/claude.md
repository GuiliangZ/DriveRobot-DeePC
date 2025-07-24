I'm a control engineer working on a drive robot. The goal of drive robot is to use a PWM signal(-100% to 100%) to interact with A/B pedal of the car to make the car track a predefined speed profile as precisely as smoothly as possible. 
In control's lanuage, I have a time variant SISO system, My single input is -100 to 100 PWM signal, and my only measured output is dyno speed from the car V_dyno. 
My system is changing slowly over time because the pedal position will generate different torque along with decreaseing state of the charge of the vehicle.
I want to use the minimum control effort to track the predefined speed profile as precisely as possible, need to account for system noise for the real-system. 
I cannot measure any internal states, but I can utilize abundant of recorded SISO data to design my controller. 
I have implement a DeePC based real-time controller - using acados and casadi.
Now I am in the parameter tunning phase, Can you design the code to help me to find the best DeePC related paramters first for best speed tracking(speed tracking error<1kph and make the vehicle drive as smooth as possible). 
Also, I just want my control frequency to be 10Hz, so you can decide if I need to use acados as real-time solver or not. May be we can try cvxpy, cvxgen for optimization solver. 
Please impelement and modify real-time DeePC controller with 10Hz loop-updating frequency by modifying the "PIDDeePCController.py" and "DeePCAcados.py".
I want to keep the structure of controller where use baselinePID as fall back controller whenever DeePC is unstable or doesn't work. 
