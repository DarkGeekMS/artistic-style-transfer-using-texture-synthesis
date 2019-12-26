# Artistic Style Transfer

This is a simple implmentation for _Style-Transfer via Texture-Synthesis_ paper in python, which is a classical method in style transfer without the use of neural networks. 

Paper: https://arxiv.org/abs/1609.03057

## Installation 

```
    pip install -r requirements.txt
```

## Usage

- For GUI:
```
   python gui.py
```

- Terminal Execution:
```
   python main.py --content_path <path_to_content> --style_path <path_to_style>
```
Specify other arguments, if needed.

## Output Sample

![Alt text](/data/content/house2small.jpg?raw=true "Content")
![Alt text](/data/style/derschrei.jpg?raw=true "Style")
![Alt text](/outputs/samples/output_2.png?raw=true "Output")

**Cheers ^ ^**