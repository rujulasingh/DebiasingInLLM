# DebiasingInLLM
Problem Statement
NLP models, especially those trained on vast amounts of text data, often inadvertently inherit biases present in that data. These biases can perpetuate stereotypes and harmfully impact applications built on top of these models.

Objectives
Modify Attention Mechanism: Instead of merely manipulating word embeddings or training data, the approach concentrates on altering the attention mechanism of transformer models.

Ensure Equal Attention: The revised attention mechanism gives equal weights to all demographic groups present in the input data. This equal distribution of attention ensures that no particular group is unduly emphasized.

Calibrate Attention on Identity Terms: The strategy seeks to decrease harmful associations by fine-tuning the attention context concerning identity-related terms.
