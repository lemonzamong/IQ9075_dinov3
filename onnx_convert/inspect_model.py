
import onnx
model = onnx.load("common/dinov3.onnx")
print("Input(s):")
for input in model.graph.input:
    print(f"Name: {input.name}, Type: {input.type}")
print("\nOutput(s):")
for output in model.graph.output:
    print(f"Name: {output.name}, Type: {output.type}")
