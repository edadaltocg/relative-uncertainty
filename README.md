# A Data-Driven Measure of Relative Uncertainty for Misclassification Detection

Checkpoints are online and available at: `https://github.com/edadaltocg/relative-uncertainty/releases/tag/checkpoints`.

## Main code

```python
import torch


class MetricLearningLagrange:
    def __init__(self, model, lbd=0.5, temperature=1, **kwargs):
        self.model = model
        self.device = next(model.parameters()).device
        self.lbd = lbd
        self.temperature = temperature

        self.params = None

    def fit(self, train_dataloader, *args, **kwargs):
        # get train logits
        train_logits = []
        train_labels = []
        for data, labels in train_dataloader:
            data = data.to(self.device)
            with torch.no_grad():
                logits = self.model(data).cpu()
            if logits.shape[1] % 2 == 1:  # openmix
                logits = logits[:, :-1]
            train_logits.append(logits)
            train_labels.append(labels)
        train_logits = torch.cat(train_logits, dim=0)
        train_pred = train_logits.argmax(dim=1)
        train_labels = torch.cat(train_labels, dim=0)
        train_labels = (train_labels != train_pred).int()

        train_probs = torch.softmax(train_logits / self.temperature, dim=1)

        train_probs_pos = train_probs[train_labels == 0]
        train_probs_neg = train_probs[train_labels == 1]

        self.params = -(1 - self.lbd) * torch.einsum("ij,ik->ijk", train_probs_pos, train_probs_pos).mean(dim=0).to(
            self.device
        ) + self.lbd * torch.einsum("ij,ik->ijk", train_probs_neg, train_probs_neg).mean(dim=0).to(self.device)
        self.params = torch.tril(self.params, diagonal=-1)
        self.params = self.params + self.params.T
        self.params = torch.relu(self.params)
        if torch.all(self.params <= 0):
            # default to gini
            self.params = torch.ones(self.params.size()).to(self.device)
            self.params = torch.tril(self.params, diagonal=-1)
            self.params = self.params + self.params.T
        self.params = self.params / self.params.norm()

    def __call__(self, logits, *args, **kwds):
        probs = torch.softmax(logits / self.temperature, dim=1)
        # double check that constraints are satisfied
        params = torch.tril(self.params, diagonal=-1)
        params = params + params.T
        params = params / params.norm()
        return torch.diag(probs @ params @ probs.T)
```

## Usage

### Misclassification

Variables:

```bash
model_names=("densenet121_cifar10" "resnet34_cifar10" "densenet121_cifar100" "resnet34_cifar100")
styles=("ce" "openmix" "mixup" "regmixup" "lognorm")
python -m src.RelU.main \
    --model_name resnet34_cifar10 \
    --r 2 \
    --style ce \
    --method metric_lagrange \
    --lbd 0.5 \
    --seed 1
```

### Corruption

```bash
# D matrix
## train
model_names=("densenet121_custom" "resnet34_custom")
tps=("" "_mixup" "_regmixup" "_lognorm")
is=(1 2 3)

python -m corruption_analysis.d_matrix.d_matrix_train_script --config_file_path ./corruption_analysis/d_matrix/corruption_d_matrix_config.yaml > ./out_files_train_corruption_d_matrix/out.log

## eval
python -m corruption_analysis.d_matrix.d_matrix_eval_script --config_file_path ./corruption_analysis/d_matrix/corruption_d_matrix_config.yaml > ./out_files_eval_corruption_d_matrix/out.log

# Doctor
python -m corruption_analysis.doctor.doctor_script_compute --config_file_path corruption_analysis/doctor/corruption_doctor_config.yaml > ./out_files_compute_corruption_doctor/out.log

## eval
python -m corruption_analysis.doctor.doctor_script_eval --config_file_path corruption_analysis/doctor/corruption_doctor_config.yaml > ./out_files_eval_corruption_doctor/out.log
```

### Mismatch

```bash
# D matrix
## train
python -m mismatch_analysis.d_matrix.d_matrix_train_script --config_file_path ./mismatch_analysis/d_matrix/mismatch_d_matrix_config.yaml > ./out_files_train_mismatch_d_matrix/out.log

## eval
python -m mismatch_analysis.d_matrix.d_matrix_eval_script --config_file_path ./mismatch_analysis/d_matrix/mismatch_d_matrix_config.yaml > ./out_files_eval_mismatch_d_matrix/out.log

# Doctor
python -m mismatch_analysis.doctor.doctor_script --model_idx ${i}
```

## Environment variables

Environment variables are set in `.env`. Run `source .env` to export them.

```
# .env
export ROOT_DIR=
export CHECKPOINTS_DIR=${ROOT_DIR}/checkpoints
export DATA_DIR=${ROOT_DIR}/data
```

## Citing this work

```
@misc{dadalto2023datadriven,
      title={A Data-Driven Measure of Relative Uncertainty for Misclassification Detection}, 
      author={Eduardo Dadalto and Marco Romanelli and Georg Pichler and Pablo Piantanida},
      year={2023},
      eprint={2306.01710},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
