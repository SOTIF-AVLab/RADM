# RADM
## Abstract
Motion prediction modules are paramount for autonomous vehicles to forecast the future behavior of surrounding road users. Failures in prediction modules can mislead the downstream planner into making unsafe decisions. Currently, deep learning technology has been widely utilized to design prediction models due to its impressive performance. However, such models may fail in the “long-tail” driving scenarios where the training data is sparse or unavailable, which is defined as the so-called epistemic uncertainty of prediction models. In this paper, we propose a safe decision-making framework to handle the epistemic uncertainty arising from training the prediction model on insufficient data. Firstly, a multi-agent prediction network with epistemic uncertainty quantification is proposed, which simultaneously takes the historical states of nearby road users, map information, and traffic lights as inputs. Then, a decision-making method based on model predictive control is designed to process not only the multi-agent prediction results but also consider the epistemic uncertainty of the prediction model. In addition, real-world driving datasets are utilized to validate the prediction model with epistemic uncertainty quantification. Furthermore, the proposed decision-making framework is evaluated via both log-replay data from real-world driving logs and the SUMO simulator where multiple challenging cases, e.g, pedestrians and non-motorized vehicles crossing the intersection illegally, are considered. Experiment results demonstrate that the proposed method could lower the driving risk when the prediction model fails.

## Experimental Results- Qualitative Performance Evaluation Via Real-World Driving Data Replay

### Case 1: Pedestrian crosses intersection illegally

#### VMPC:

https://user-images.githubusercontent.com/52814283/221113009-11c351ba-8a4a-4b99-b4a0-b3c2ca083540.mp4

#### RADM:

https://user-images.githubusercontent.com/52814283/221112342-6f44e3b4-257d-446d-b77a-b05a9d68b70b.mp4

### Case 2: Non-motorized vehicle turns left illegally

####VMPC:

https://user-images.githubusercontent.com/52814283/216960186-a322327e-d426-409c-956b-701d51aae963.mp4

####RADM:

https://user-images.githubusercontent.com/52814283/216960250-8e3c4f89-ef68-4fc1-b7a5-7989d5d075cb.mp4

###Case 3: Non-motorized vehicle making the U-Turn illegally

#### VMPC:

https://user-images.githubusercontent.com/52814283/216960378-f852976f-0d93-4a4a-8983-785e3df77bfa.mp4

#### RADM:

https://user-images.githubusercontent.com/52814283/216960436-7f45ecff-70d3-44c7-945a-61a9c73d67ba.mp4


