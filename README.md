# Инструкция по воспроизводимости

- Cuda driver:
  - Cuda compilation tools, release 11.1, V11.1.74
  - Build cuda_11.1.TC455_06.29069683_0

- Dependencies: `requirements.txt`

- ONNX conversion: 
  - `onnx==1.15.0`
  - `onnxruntime==1.16.3`
  - opset_version: 12 
  - input_names: 
    - INPUT_IDS
    - ATTENTION_MASK
  - output_names:
    - EMBEDDINGS
  - dynamic_axes:
    - INPUT_IDS: 
      - 0: BATCH_SIZE
    - ATTENTION_MASK: 
      - 0: BATCH_SIZE
    - EMBEDDINGS:
      - 0: BATCH_SIZE
     
Script for onnx convertation: `./export_model.py`  
Torch model path: `./model.pt`



