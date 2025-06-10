## Modifications to G.pt Source Code

We document the minimal changes made to the evaluated method's source code (under [method](method)):
1. [Gpt/download.py](method/Gpt/download.py): instead of downloading all weight datasets and pre-trained checkpoints, only download MNIST classification model weights and corresponding pre-trained checkpoint.

```diff
- pretrained_models = {'mnist_loss.pt', 'mnist_error.pt', 'cartpole.pt', 'cifar10_loss.pt', 'cifar10_error.pt'}
- checkpoint_datasets = {'mnist', 'cartpole', 'cifar10'}
+ pretrained_models = {'mnist_loss.pt'}
+ checkpoint_datasets = {'mnist'}
```

2. [data_gen/train_mnist.py](method/data_gen/train_mnist.py): implement ``test_accs_and_incorrect_indices``, which returns the accuracy and the test images the input model predicts incorrectly.

```diff
+ @torch.inference_mode()
+ def test_accs_and_incorrect_indices(inputs, labels, model):
+     model.eval()
+     preds = model(inputs)
+     top1_err, _ = top1_error(preds, labels)
+     incorrect_indices = preds.argmax(dim=1) != labels
+     return 100 - top1_err, incorrect_indices
```

3. [Gpt/tasks.py](method/G.pt/method/Gpt/tasks.py): add the implemented ``test_accs_and_incorrect_indices`` function to ``TASK_METADATA``.

```diff
     "mnist_loss": {
         "task_test_fn": data_gen.train_mnist.test_epoch,
+        "test_accs_and_incorrect_indices": data_gen.train_mnist.test_accs_and_incorrect_indices,
         "constructor": lambda: data_gen.train_mnist.MLP(w_h=10),
         "data_fn": data_gen.train_mnist.unload_test_set,
         "aug_fn": data_gen.train_mnist.random_permute_mlp,
         "minimize": True,
         "best_prompt": 0.0,
         "recursive_prompt": 0.0
     },
```