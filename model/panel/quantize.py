"""
@author: Lochana Marasinghe
@date: 2/27/2026
@description:
 .pt to onnx conversion command:
 yolo export model=model/panel/yolo12s_custom_panel_detection_combined.pt format=onnx  device=CPU simplify=True

 then do preprocess before quantization
 python -m onnxruntime.quantization.preprocess --input model/panel/yolo12s_custom_panel_detection_combined.onnx --output model/panel/yolo12s_custom_panel_detection_combined_preprocessed.onnx

"""
from onnxruntime.quantization import quantize_dynamic, QuantType


input_model = "yolo12s_custom_panel_detection_combined_preprocessed.onnx"
output_model = "yolo12s_custom_panel_detection_combined_int8.onnx"

quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QInt8  # INT8 weights
)

print("Quantized model saved:", output_model)