# Flip Operator

Mirror each record left to right or top to bottom. The flip is deterministic; torchvision's
`RandomHorizontalFlip(p)` is `ProbabilisticOperator(probability=p)` around a horizontal
`FlipOperator`, which decides per record and passes records through in eval mode.

## See Also

- [Operators Overview](index.md) - All operator types
- [Probabilistic Operator](probabilistic_operator.md) - Apply an operator with a probability
- [Random Crop Operator](random_crop_operator.md) - Pad and crop at a random offset
- [Operator Cheat Sheet](../examples/quick-reference/operator-cheatsheet.md#image-operators)

---

::: datarax.operators.modality.image.flip_operator
