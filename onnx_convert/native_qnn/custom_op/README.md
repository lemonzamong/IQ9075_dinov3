# Custom Op Package for QNN

This directory is intended for Custom Operator Packages.
If `qnn-onnx-converter` reports an Unsupported Op, you must identify the Op type and create a package here.

## Structure
- `src/`: Source code for the operations (C++/ASM)
- `config/`: Configuration XML for the package generator.

## Usage
1. Use `qnn-op-package-generator` to create the package skeleton.
   ```bash
   qnn-op-package-generator -p MyCustomOpPackage -o .
   ```
2. Implement the kernel in the generated files.
3. Compile using the SDK tools.

Note: Currently, DINOv3 conversion passed for 224x224 input without custom ops. If a specific op fails in the future, follow this guide.
