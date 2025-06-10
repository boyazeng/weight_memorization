## Modifications to P-diff Source Code

We document the minimal changes made to the evaluated method's source code (under [method](method)):
1. Modify root path to account for the change of source code folder name from "Neural-Network-Diffusion" to "method".

```diff
- root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Neural-Network-Diffusion")+1])
+ root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("method")+1])
```

2. [workspace/run_all.sh](method/workspace/run_all.sh): we remove the call to [workspace/evaluate.py](method/workspace/evaluate.py) since we use our own evaluation script.

```diff
  cd "../../../workspace" || exit 
  bash launch.sh "$cls" "$tag" "$device" 
  CUDA_VISIBLE_DEVICES="$device" python generate.py "$cls" "$tag" 
- CUDA_VISIBLE_DEVICES="$device" python evaluate.py "$cls" "$tag"
  cd ..
```