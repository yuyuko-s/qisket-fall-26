- QuantumFeatureTransformer
Converts input feature vector into quantum measurement features
 - How?
 - Inputs are encoded into phases (Angle encoding). We use 
 It takes the input features from the PCA
It uses Qiskit's zz_feature_map to encode the inputs into phases (angle encode), which converts the input features into a quantum feature. 
The quantum features are used as an input table for classical ridge regression
