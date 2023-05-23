# Title-Prediction Failure Risk-Aware Decision-Making for Autonomous Vehicles on Signalized Intersections
## Abstract
Motion prediction modules are crucial for autonomous vehicles to forecast the future behavior of surrounding road users. Failures in prediction modules can mislead a downstream planner to make unsafe decisions. Currently, deep learning technology has been widely used to design prediction models due to its impressive performance. However, such models may fail in long-tail driving scenarios where the training data are insufficient or unavailable, which represents the so-called epistemic uncertainty of prediction models. This paper proposes a risk-aware decision-making (RADM) framework to handle the epistemic uncertainty arising from training the prediction model on insufficient data. First, a multi-agent prediction network with epistemic uncertainty quantification is proposed. This network uses the historical states of nearby road users, map information, and traffic lights as inputs. Then, the RADM utilizes model predictive control technique to not only process the multi-agent prediction results but also to consider the epistemic uncertainty of the prediction model. In addition, the accuracy of the established prediction model is verified on real-world driving datasets. Furthermore, the proposed RADM is evaluated on the log-replay data obtained from real-world driving logs and using the SUMO simulator, considering multiple challenging cases where pedestrians and non-motorized vehicles cross the intersection illegally. The experimental results demonstrate that RADM can reduce the driving risk and improve driving safety and supplementary videos are provided.

## Experimental Results- Qualitative Performance Evaluation Via Real-World Driving Data Replay

### Case 1: Pedestrian crosses intersection illegally
VMPC |　RADM
:-------------------------:|:-------------------------:
https://github.com/SOTIF-AVLab/RADM/assets/52814283/1a7230bd-1eb6-4476-b421-db08bb3744d6 | https://github.com/SOTIF-AVLab/RADM/assets/52814283/30958c14-f4d8-438a-863e-0ca3b2b08b00

### Case 2: Non-motorized vehicle turns left illegally
VMPC |　RADM
:-------------------------:|:-------------------------:
https://github.com/SOTIF-AVLab/RADM/assets/52814283/785547ad-171f-4f0b-928d-4c7edf82d8af | https://github.com/SOTIF-AVLab/RADM/assets/52814283/b1496bd0-6e9e-4de4-b1ac-e7b22bd26ad2

### Case 3: Non-motorized vehicle making the U-Turn illegally

VMPC |　RADM
:-------------------------:|:-------------------------:
https://github.com/SOTIF-AVLab/RADM/assets/52814283/c6e6f550-4cca-4607-aeb6-a05c0676afaf | https://github.com/SOTIF-AVLab/RADM/assets/52814283/59fc1fdf-5aec-462c-a1c9-d7848e4f0a3b


## Experimental Results- Quantitive Performance Evaluation by SUMO Simulator
### Challenging Scenarios-RADM
<!-- 
https://github.com/SOTIF-AVLab/RADM/assets/52814283/56dc1d47-ecf3-46ea-8b0a-a8d037297a4b

https://github.com/SOTIF-AVLab/RADM/assets/52814283/ab671b60-2bcd-4f44-b15c-383b24f190b0 -->

#### Case 1-Pedestrians crossing intersection illegally
https://github.com/SOTIF-AVLab/RADM/assets/52814283/a4e73248-6f82-4caa-8c6d-6ea4b45c6133

#### Case 2-Pedestrians crossing intersection illegally
https://github.com/SOTIF-AVLab/RADM/assets/52814283/649fdc3b-07e9-4762-b8b0-2362e934098f

#### Case 3-Pedestrians crossing intersection illegally
https://github.com/SOTIF-AVLab/RADM/assets/52814283/9d666bac-6544-4d36-ba37-644d6e7d3edf

#### Case 4-Non-motorized vehicles crossing intersection illegally
https://github.com/SOTIF-AVLab/RADM/assets/52814283/5d14d2a8-c0e2-4a67-920c-6cd5c6c91d7d

#### Case 5-Non-motorized vehicles crossing intersection illegally
https://github.com/SOTIF-AVLab/RADM/assets/52814283/114c354a-2629-44d6-b301-81e87d89a08e
