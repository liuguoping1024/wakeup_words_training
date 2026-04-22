#!/usr/bin/env python3
from ai_edge_litert.interpreter import Interpreter
import sys

for model in sys.argv[1:]:
    interp = Interpreter(model_path=model)
    interp.allocate_tensors()
    print(f"\n{model}:")
    for d in interp.get_input_details():
        print(f"  input[{d['index']}]: shape={d['shape']}, dtype={d['dtype']}, name={d['name']}")
    for d in interp.get_output_details():
        print(f"  output[{d['index']}]: shape={d['shape']}, dtype={d['dtype']}, name={d['name']}")
