# VertexWeight
**Python** implementation of &lt;**Exploring High-Order Correlations for Industry Anomaly Detection**>

Paper: Exploring High-Order Correlations for Industry
Anomaly Detection <br/>

**IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS**

**Abstract**—Anomaly detection aims to find the outlier
noise data, which has attracted much attention in recent
years. However, in most industrial cases, we can only obtain very few labeled anomalous data with abundant unlabeled data, which makes anomaly detection intractable. To
tackle this issue, we propose a vertex-weighted hypergraph
structure (VWHL) and a learning algorithm on it for anomaly
detection. In this method, the correlation among data is
formulated in a hypergraph structure, where each vertex
denotes one sample, and the connections (hyperedges) on
the hypergraph represent the relation among samples in
feature space. We introduce vertex weights to investigate
the importance of different samples, where each vertex
is initialized with a weight corresponding to its similarity
score and isolation score. Then, we learn the label projection matrix for anomaly detection and optimize vertex
weights simultaneously. In this way, the vertices with high
weights, which indicate data likely to be anomalies, play an
important role during the learning process. We have evaluated our method on Industry Anomaly Detection dataset,
Outlier Detection Datasets (ODDS) dataset and Software
Defect Prediction (SDP) Dataset, and experimental results
show that our method achieves better performance when
compared with other state-of-the-art methods.