trtexec --onnx=/model/1/model.onnx \
--saveEngine=/model/1/model.plan \
--minShapes=INPUT_IDS:1x16,ATTENTION_MASK:1x16 \
--optShapes=INPUT_IDS:8x16,ATTENTION_MASK:8x16 \
--maxShapes=INPUT_IDS:8x16,ATTENTION_MASK:8x16 \
--fp16 \
--verbose=True

