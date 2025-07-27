# Pre-existing Test Failures in Luminal

As of the legacy_prims refactoring PR, the following tests are failing. These failures appear to be pre-existing issues unrelated to the legacy_prims changes:

## Failed Tests

### 1. Arange-related tests (returning NaN):
- `hl_ops::other::tests::test_arange`
- `hl_ops::other::tests::test_arange_from_zero`
- `hl_ops::other::tests::test_arange_in_range`
- `hl_ops::other::tests::test_arange_step_fractional`
- `hl_ops::other::tests::test_arange_step_simple`
- `hl_ops::other::tests::test_dyn_arange`

**Issue**: All arange functions return NaN values instead of the expected sequential numbers.

### 2. Triangular matrix tests:
- `hl_ops::other::tests::test_tril`
- `hl_ops::other::tests::test_triu`

**Issue**: Returns 0.0 instead of 1.0 at expected positions.

### 3. Gather test:
- `hl_ops::other::tests::test_gather`

**Issue**: Returns NaN instead of expected values.

## Notes
- These tests fail both with and without the `legacy_prims` feature enabled
- The failures are likely related to the cumsum implementation or how reductions are being compiled
- This should be investigated in a separate PR focused on fixing these specific issues 