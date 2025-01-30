
# Run batched

This is a very basic module that allows you to run a pytorch model on numpy arrays in batches. It handles
the batching for you, so you can just pass in a numpy array and get a numpy array back. It also works when the
return value is a tuple/list/dict of numpy arrays.