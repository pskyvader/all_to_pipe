# All to Pipe

ComfyUI custom nodes to build, manipulate, and export a reusable generation pipeline
("pipe") containing models, LoRAs, parameters, and prompt structures.

This module is intentionally explicit:

- No default values are set anywhere
- All data is strongly structured via classes
- Encoding and sampling are deliberately separated
- Export paths are split between sampler connections and plain data / JSON
- all classes, methods, variables, etc must be typed. (strictly required)
