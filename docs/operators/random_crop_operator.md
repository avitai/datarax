# Random Crop Operator

Pad each record and crop it at its own random offset, as torchvision's `RandomCrop`: padding in
torchvision's three forms and four modes (constant, edge, reflect, symmetric), offsets uniform
over every valid position. Eval mode takes the centre crop of the padded record.

## See Also

- [Operators Overview](index.md) - All operator types
- [Flip Operator](flip_operator.md) - Mirror records left to right or top to bottom
- [Functional API](functional.md) - `pad`, `center_crop`, `random_crop`
- [Operator Cheat Sheet](../examples/quick-reference/operator-cheatsheet.md#image-operators)

---

::: datarax.operators.modality.image.random_crop_operator
