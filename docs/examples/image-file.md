# Image File

Classifies one photo and acts on the label only when the model is confident enough. Put a `photo.jpg` next to the model files.

## Code

```python title="examples/image_file.py"
--8<-- "examples/image_file.py"
```

## How it works

- `class_name` includes the class number, so the script removes it.
- `class_confidence` goes from 0 to 1, so `0.8` means 80%.
