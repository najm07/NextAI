# GPU Training Performance Optimizations

## Why Training Was Slow

The original implementation had several bottlenecks:

1. **No DataLoader**: Processing data in Python loops is CPU-bound
2. **Inefficient tensor creation**: Creating tensors one-by-one in loops
3. **No mixed precision**: Not using FP16 which can be 2x faster
4. **Small batch size**: Default 64 is too small for T4 GPU (16GB)
5. **CPU-GPU transfer overhead**: Moving data in training loop

## Optimizations Applied

### 1. PyTorch DataLoader
- **Before**: Manual batching in Python loops
- **After**: `DataLoader` with `num_workers=4` for parallel data loading
- **Speedup**: ~2-3x faster data loading

### 2. Mixed Precision Training (FP16)
- **Before**: Full FP32 precision
- **After**: Automatic Mixed Precision (AMP) with FP16
- **Speedup**: ~2x faster on T4 GPU, uses less memory

### 3. Larger Batch Size
- **Before**: `batch_size=64`
- **After**: `batch_size=512` (or even 1024 for T4)
- **Speedup**: Better GPU utilization, fewer kernel launches

### 4. Efficient Tensor Batching
- **Before**: Creating tensors one-by-one in loops
- **After**: Batch tensor operations with masking
- **Speedup**: Eliminates CPU-GPU transfer overhead

### 5. Pin Memory
- **Before**: Standard memory allocation
- **After**: `pin_memory=True` for faster CPU-GPU transfers
- **Speedup**: ~10-20% faster data transfer

## Expected Performance Improvements

- **Overall speedup**: 4-6x faster training
- **Memory usage**: ~50% reduction with FP16
- **GPU utilization**: Should be 80-95% (was probably 20-30%)

## Usage

Update your training call:

```python
trained_model = train_parser_gpu(
    target_accuracy=0.95,
    max_epochs=100,
    lr=0.001,
    batch_size=512,  # Increased from 64
    n_train=500000,
    n_val=50000,
    seed=42,
    corpus=corpus_path,
    device=device,
    use_amp=True,      # NEW: Mixed precision
    num_workers=4,     # NEW: Parallel loading
    pin_memory=True    # NEW: Faster transfers
)
```

## Monitoring GPU Usage

Check GPU utilization in Colab:
```python
!nvidia-smi
```

You should see:
- **GPU utilization**: 80-95% (was probably 20-30%)
- **Memory usage**: ~2-4GB (was probably <1GB)
- **Power draw**: Higher (means GPU is working)

## Further Optimizations (if still slow)

1. **Increase batch size to 1024** if you have memory:
   ```python
   batch_size=1024
   ```

2. **Use gradient accumulation** for very large effective batches:
   ```python
   # In training loop
   accumulation_steps = 2
   loss = loss / accumulation_steps
   loss.backward()
   if (batch_idx + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```

3. **Reduce validation frequency**:
   ```python
   if epoch % 5 == 0:  # Validate every 5 epochs instead of every epoch
       # validation code
   ```

4. **Use torch.compile()** (PyTorch 2.0+):
   ```python
   model = torch.compile(model)  # JIT compilation
   ```

