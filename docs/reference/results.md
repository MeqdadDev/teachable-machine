# Results

`classify_image` returns a dictionary like this one, for a model with four classes:

```python
{
    "class_name": "0 Red",          # the line from labels.txt
    "highest_class_name": "0 Red",
    "highest_class_id": 0,
    "class_index": 0,               # position of the class, starting at 0
    "class_id": 0,
    "predictions": array([0.9319, 0.0003, 0.0003, 0.0675], dtype=float32),
    "all_predictions": array([0.9319, 0.0003, 0.0003, 0.0675], dtype=float32),
    "class_confidence": 0.9319,     # from 0 to 1
    "highest_class_confidence": 0.9319,
}
```

| Key | Type | Meaning |
|---|---|---|
| `class_name` | `str` | Line of `labels.txt` for the predicted class, number included |
| `highest_class_name` | `str` | Same as `class_name` |
| `class_index` | `numpy.int64` | Position of the predicted class, starting at 0 |
| `class_id`, `highest_class_id` | `numpy.int64` | Same as `class_index` |
| `class_confidence` | `numpy.float32` | Confidence from 0 to 1 |
| `highest_class_confidence` | `numpy.float32` | Same as `class_confidence` |
| `predictions` | NumPy array | One value per class, in the order of `labels.txt` |
| `all_predictions` | NumPy array | Same as `predictions` |

To show a percentage:

```python
print(f"{result['class_confidence']:.1%}")  # 93.2%
```

`classify_and_show` returns a tuple of this dictionary and the annotated image.

## Convert to JSON

`json.dumps` can't handle NumPy values, so convert the ones you need:

```python
import json

data = {
    "class_name": result["class_name"],
    "class_index": int(result["class_index"]),
    "class_confidence": float(result["class_confidence"]),
    "predictions": result["predictions"].tolist(),
}
print(json.dumps(data))
```
