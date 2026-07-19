**This is a completed implementation of a basic Transformer language model coded using just numpy. Please see proof_of_concept.ipynb for a demo of the working code**

I am coding my way through deep learning history using just numpy to improve my understanding of AI. I will use it as a base off of which to continue learning by building other deep learning innovations (e.g., improvements to the Transformer like RoPE, Adam, better memory management; RL post-training).

I am using this doc to detail each step of the forward and backward pass to make sure I am truly understanding what I am building: https://docs.google.com/spreadsheets/d/1bDB1wPxcVrq55hjSDb3glXPYFl0f3-6sNImfGcF9huw/edit?usp=sharing

No AI-generated code was used in building the Transformer. The only AI-generated code in this repo is in the quick_testing notebook. I did use AI as a teacher - to validate my understanding or answer questions that arose while reading through papers and articles.

The final version here uses CuPy, so it cannot be run on a CPU. I ran code by connecting to a Colab GPU runtime from my IDE.