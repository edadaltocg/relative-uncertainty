import torch
import numpy as np
from sklearn.metrics import auc, roc_curve, average_precision_score
from tqdm import tqdm
from torch.autograd import Variable
from torch_uncertainty.metrics.classification import AURC
import pandas as pd
from code.utils.helper import _prepare_config_for_results
from code.utils.metrics import compute_all_metrics



def selective_net_risk(scores, pred, targets, thr: float):

    covered_idx = scores <= thr

    return np.sum(pred[covered_idx] != targets[covered_idx]) / np.sum(covered_idx)

def hard_coverage(scores, thr: float):
    return (scores <= thr).mean()

def risks_coverages_selective_net(scores, pred, targets, sort=True):
    """
    Returns:

        risks, coverages, thrs
    """
    # this function is slow
    risks = []
    coverages = []
    thrs = []
    for thr in np.unique(scores):
        risks.append(selective_net_risk(scores, pred, targets, thr))
        coverages.append(hard_coverage(scores, thr))
        thrs.append(thr)
    risks = np.array(risks)
    coverages = np.array(coverages)
    thrs = np.array(thrs)

    # sort by coverages
    if sort:
        sorted_idx = np.argsort(coverages)
        risks = risks[sorted_idx]
        coverages = coverages[sorted_idx]
        thrs = thrs[sorted_idx]
    return risks, coverages, thrs

# class DetectorEvaluator:
#     def __init__(self, model, dataloader, device, magnitude = 0, 
#                  return_embs=False, return_labels=False, return_model_preds=False,
#                  path=None):
#         """
#         Evaluator for measuring model accuracy.
        
#         Args:
#             model (nn.Module): The trained classifier.
#             dataloader (DataLoader): DataLoader for evaluation data.
#             device (torch.device): Device to perform evaluation on.
#         """
#         self.model = model.to(device)
#         self.dataloader = dataloader
#         self.device = device
#         self.return_embs = return_embs
#         self.return_labels = return_labels  # Whether to return labels or not
#         self.return_model_preds = return_model_preds
#         self.path = path
#         self.magnitude = magnitude

#     def fpr_at_fixed_tpr(self, fprs, tprs, thresholds, tpr_level: float = 0.95):
        
#         idxs = [i for i, x in enumerate(tprs) if x >= tpr_level]
#         if len(idxs) > 0:
#             idx = min(idxs)
#         else:
#             idx = 0
#         return fprs[idx], tprs[idx], thresholds[idx]

#     def evaluate(self, detector, return_clusters=False):

#         self.model.eval()

#         all_clusters = []
#         all_embs = []
#         all_detector_preds = []
#         all_detector_labels = []
#         all_labels = []
#         all_model_preds = []
#         aurc = AURC()
        
#         for inputs, labels in tqdm(self.dataloader, desc="Evaluating Detector", disable=False):
#             inputs = inputs.to(self.device)
#             labels = labels.to(self.device)
#             # x: [batch_size, dim], labels: [batch_size]


#             if self.magnitude > 0:
                
#                 inputs = Variable(inputs, requires_grad=True)
#                 logits = self.model(inputs)
#                 model_preds = torch.argmax(logits, dim=1)
#                 detector_labels = model_preds != labels

#                 detector_preds = detector(logits=logits)
#                 torch.log(detector_preds).sum().backward()
          
#                 inputs = inputs - self.magnitude * torch.sign(-inputs.grad)
#             # inputs = torch.clamp(inputs, 0, 1)
#                 inputs = Variable(inputs, requires_grad=False)

#                 with torch.no_grad():
#                     detector_preds = detector(inputs)

#             else:
#                 with torch.no_grad():
#                     logits = self.model(inputs)  # logits: [batch_size, num_classes]
#                     model_preds = torch.argmax(logits, dim=1)  # [batch_size]
            
#                     detector_labels = model_preds != labels
#                     detector_preds = detector(inputs)
            


#             if return_clusters:
#                 clusters = detector.predict_clusters(inputs)
#                 # embs = detector.feature_extractor(inputs).squeeze(-1)
#             else:
#                 clusters = torch.tensor([np.nan] * inputs.shape[0], device=self.device) # [np.nan] * inputs.shape[0]
#                 # embs = torch.tensor([np.nan] * inputs.shape[0], device=self.device) # [np.nan] * inputs.shape[0]
#             if self.return_embs:
#                 embs = detector.feature_extractor(inputs).squeeze(-1)
#             else:
#                 embs = torch.tensor([np.nan] * inputs.shape[0], device=self.device)
#             # if not self.return_labels:
#             #     labels = torch.tensor([np.nan] * inputs.shape[0], device=self.device)
#             # if not self.return_model_preds:
#             #     model_preds = torch.tensor([np.nan] * inputs.shape[0], device=self.device)
          
#             # aurc.update(detector_preds, detector_labels)
            
        
#             all_model_preds.append(model_preds.cpu().numpy())
#             all_clusters.append(clusters.cpu().numpy())
#             all_embs.append(embs.cpu().numpy())
#             all_detector_labels.append(detector_labels.cpu().numpy())
#             all_detector_preds.append(detector_preds.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())

#         all_detector_preds = np.concatenate(all_detector_preds, axis=0)
#         all_detector_labels = np.concatenate(all_detector_labels, axis=0)
#         all_clusters = np.concatenate(all_clusters, axis=0)
#         all_embs = np.concatenate(all_embs, axis=0)
#         all_labels = np.concatenate(all_labels, axis=0)
#         all_model_preds = np.concatenate(all_model_preds, axis=0)
#         # print("aurc_result", aurc.compute())
#         if self.path is not None:
#             np.savez_compressed(
#                 self.path,
#                 embs=all_embs,
#                 detector_preds=all_detector_preds,
#                 detector_labels=all_detector_labels,
#                 clusters=all_clusters,
#                 labels=all_labels,
#                 model_preds=all_model_preds
#             )

#         fprs, tprs, thrs = roc_curve(all_detector_labels, all_detector_preds)
#         # Compute the area under the ROC curve
#         roc_auc = auc(fprs, tprs)
#         fpr, tpr, thr = self.fpr_at_fixed_tpr(fprs, tprs, thrs, 0.95)
#         model_acc = (all_model_preds == all_labels).mean()
#         risks, coverages, _ = risks_coverages_selective_net(
#             np.squeeze(all_detector_preds),
#             np.squeeze(all_model_preds), 
#             all_labels)
        
        
#         # prepend (0, 0) if missing
#         if coverages.size == 0 or coverages[0] > 0.0:
#             coverages = np.r_[0.0, coverages]
#             risks = np.r_[0.0, risks]

#         # append (1, overall_err) if missing
#         if coverages[-1] < 1.0:
#             overall_err = (all_model_preds != all_labels).mean()
#             coverages = np.r_[coverages, 1.0]
#             risks = np.r_[risks, overall_err]
#         aurc = auc(coverages, risks)
    
#         aupr_err = average_precision_score(all_detector_labels, all_detector_preds)
#         aupr_success = average_precision_score(1 - all_detector_labels, 1 - all_detector_preds)

#         return fpr, tpr, thr, roc_auc, model_acc, aurc, aupr_err, aupr_success
#         # if self.return_embs:
#         #     return fpr, tpr, thr, all_detector_preds, all_detector_labels, all_clusters, all_embs
#         # else:
#         #     return fpr, tpr, thr, all_detector_preds, all_detector_labels, [None] * len(all_detector_labels), [None] * len(all_detector_labels)



# from .datasets import DATA_INFO

# class OODEvaluator:
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         in_name,    
#         out_dataloader: torch.utils.data.DataLoader,
#         device: torch.device,
#         suffix = "train",
#         base_config=None,

#     ):
#         """
#         Evaluate a single model against multiple post-hoc detectors.

#         Args:
#             model:                A trained classifier (outputs raw logits).
#             dataloader:           DataLoader for the evaluation set.
#             device:               torch.device to run model & detectors on.
#             magnitude:            If >0, craft a one‐step adversarial example
#                                   against each detector as in your original.
#         """
#         self.model = model
#         self.in_name = in_name
#         self.device = device
#         self.suffix = suffix
#         self.base_config = base_config
#         self.method_name = base_config['method_name']
#         self.num_classes = base_config['data']['n_classes']
#         self.scores = None  # Precomputed scores, if any
#         self.root = f"storage_ood_latent/{base_config['data']['name']}_{base_config['model']['name']}/"
#         if suffix == "val":
#             self.latent_path = self.root + f"logits_{suffix}_n-epochs{base_config['data']['n_epochs']}_transform-{base_config['data']['transform']}.pt"
#         else:
#             self.latent_path = self.root + f"logits_{suffix}.pt"

#     def get_pertubated_scores(self, inputs, detector, magnitude):

#         # inputs = inputs.to(self.device)
#         inputs = inputs.to(self.device).detach().requires_grad_(True) 

#         # inputs = Variable(inputs.clone(), requires_grad=True)
#         adv_logits = self.model(inputs)
#         scores = detector(logits=adv_logits)           # initial
#         # backprop on log-score
#         loss = torch.log(scores + 1e-12).sum()
#         # loss = torch.log(scores.clamp_min(1e-12)).sum()
#         # loss.backward()
#         # step and detach
#         grad_inputs, = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)
#         with torch.no_grad():
#             adv = inputs + magnitude * grad_inputs.sign()
#         # inputs = (inputs + magnitude * inputs.grad.sign()).detach()
#         # inputs = Variable(inputs, requires_grad=False)
#         with torch.inference_mode():
#             logits_adv = self.model(adv)
#             scores_adv = detector(logits=logits_adv)
#         # with torch.no_grad():
#         #     scores_adv = detector(inputs=inputs)

#         return scores_adv


    
    
    
    

#     def get_in_scores(self, detectors, n_samples, list_configs, logits=None):

#         n_det = len(detectors)
#         latent_path = 
#         if os.path.exists(self.latent_path):

#             pkg = torch.load(self.latent_path, map_location="cpu")
#             logits = pkg["logits"].to(torch.float32).to(self.device)        # (N, C)
#             all_labels = pkg["labels"].numpy()                  # (N,)
#             all_model_preds  = pkg["model_preds"].numpy()             # (N,)
#             detector_labels_arr = (all_model_preds != all_labels)  # bool array

#             if self.method_name in ["gini", "max_proba", "metric_learning"]:

#                 self.model.to(self.device)
#                 self.model.eval()
#                 all_scores = [np.zeros(n_samples, dtype=float) for _ in range(n_det)]
#                 for idx, dec in tqdm(enumerate(detectors), total=len(detectors), desc="Getting Detectors Scores", leave=False):
#                     magnitude = list_configs[idx][self.method_name]['magnitude']
                    
#                     if magnitude > 0:
                        
#                         self.model.eval()
#                         write=0
                        
#                         for inputs, labels in self.loader:
#                             bs = inputs.size(0)
#                             scores = self.get_pertubated_scores(inputs, dec, magnitude)
#                             if isinstance(scores, torch.Tensor):
#                                 scores = scores.detach().cpu().numpy()
#                             all_scores[idx][write:write+bs] = scores
#                             write += bs
#                     else:
#                         with torch.no_grad():
#                             scores = dec(logits=logits).cpu().numpy()
#                         all_scores[idx][:] = scores


#             else:
#                 all_scores = [detector(logits=logits).cpu().numpy() for detector in detectors]

#         else:
            
#             all_scores = [np.zeros(n_samples, dtype=float) for _ in range(n_det)]
#             all_labels = np.zeros(n_samples, dtype=int)
#             all_model_preds = np.zeros(n_samples, dtype=int)
#             detector_labels_arr = np.zeros(n_samples, dtype=bool)
#             all_logits = np.zeros((n_samples, self.num_classes), dtype=float)
            

#             # iterate once over data
#             idx = 0

#             self.model.to(self.device)
#             self.model.eval()
#             for inputs, labels in tqdm(self.loader, desc="Getting Detectors Scores", leave=False):
#                 bs = inputs.size(0)
#                 inputs = inputs.to(self.device)             
#                 labels = labels.to(self.device)

#                 # forward pass for model accuracy & labels
#                 with torch.no_grad():
#                     logits = self.model(inputs)
#                     model_preds = torch.argmax(logits, dim=1)


#                 det_lab = (model_preds != labels).cpu().numpy()
#                 detector_labels_arr[idx: idx+bs] = det_lab
#                 all_labels[idx: idx+bs] = labels.cpu().numpy()
#                 all_model_preds[idx: idx+bs] = model_preds.cpu().numpy()
#                 all_logits[idx: idx+bs] = logits.cpu().numpy()

#                 # now each detector
#                 for i, det in enumerate(detectors):
#                     # -- optionally craft 1‑step adv example per detector
#                     if self.method_name in ["gini", "max_proba", "metric_learning"]:
#                         magnitude = list_configs[i][self.method_name]['magnitude']
#                         if magnitude > 0:
#                             scores = self.get_pertubated_scores(inputs, det, magnitude)
#                         else:
#                             with torch.no_grad():
#                                 scores = det(logits=logits)
#                     else:
#                         with torch.no_grad():
#                             scores = det(logits=logits)
#                     if isinstance(scores, torch.Tensor):
#                         scores = scores.cpu().numpy()
#                     all_scores[i][idx: idx+bs] = scores
                    
#                 idx += bs

#             # AFTER (robust)
#             parent = os.path.dirname(self.latent_path)
#             os.makedirs(parent, exist_ok=True)

#             if self.latent_path is not None:
#                 tmp = self.latent_path + ".tmp"
#                 torch.save(
#                     {
#                         "logits": torch.tensor(all_logits),     # compact on disk
#                         "labels": torch.tensor(all_labels).to(torch.int64),
#                         "model_preds": torch.tensor(all_model_preds).to(torch.int64),
#                     },
#                     tmp,
#                 )
#                 os.replace(tmp, self.latent_path)  # atomic rename
#         # print("all_scores shape:", all_scores.shape)
#         # print("detector_labels_arr shape:", detector_labels_arr.shape)
#         self.scores = {
#             "scores": all_scores,
#             "detector_labels": detector_labels_arr
#         } 

#     def evaluate(
#             self, 
#             list_configs, 
#             detectors=None, 
#             all_scores=None, 
#             detector_labels=None, 
#             suffix=None) -> list[dict]:
#         """
#         Run the model once per batch, then each detector on those inputs.

#         Args:
#             detectors:  Either a list of detector‐objects or a dict name→detector.
#                         Each detector must be callable as:
#                             scores = detector(inputs=..., logits=...)
#                         and return a 1‐D tensor of “outlier scores” (higher = more likely error).

#         Returns:
#             results: dict mapping detector_name → metrics dict, e.g.
#                 {
#                   "ODIN": {
#                       "fpr@95tpr": 0.12,
#                       "roc_auc": 0.94,
#                       "aupr_err": 0.88,
#                       "aupr_in": 0.90,
#                       "aurc": 0.23,
#                       "model_acc": 0.79,  # same for all detectors
#                   },
#                   …
#                 }
#         """
#         # normalize detectors into an ordered dict name→detector

#         # storage
#         n_samples = len(self.loader.dataset)
#         if all_scores is None:
#             self.get_scores(detectors, n_samples, list_configs)
#             all_scores = self.scores["scores"]
#             detector_labels = self.scores["detector_labels"]  # bool array
     
#         list_results = []
#         for i, scores in enumerate(all_scores):
        
#             fpr, tpr, thr, auroc, accuracy, aurc_value, aupr_in, aupr_ood = compute_all_metrics(
#                 conf=scores,
#                 detector_labels=detector_labels,
#             )
        
#             results = pd.DataFrame([{
#                 "fpr": fpr,
#                 "tpr": tpr,
#                 "thr": thr,
#                 "roc_auc": auroc,
#                 "model_acc": accuracy,
#                 "aurc": aurc_value,
#                 "aupr_in": aupr_in,
#                 "aupr_ood": aupr_ood,
#             }])
            
#             suffix = suffix if suffix is not None else self.suffix
#             results.columns = [f"{col}_{suffix}" for col in results.columns]
            
#             config = _prepare_config_for_results(list_configs[i])
#             config = pd.json_normalize(config, sep="_")
#             results = pd.concat([config, results], axis=1)
#             list_results.append(results)

#         return list_results

class AblationDetector:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        suffix = "train",
        latent_path=None,
        postprocessor_name="clustering",
        cfg_dataset=None,
        result_folder=None,

    ):
        """
        Evaluate a single model against multiple post-hoc detectors.

        Args:
            model:                A trained classifier (outputs raw logits).
            dataloader:           DataLoader for the evaluation set.
            device:               torch.device to run model & detectors on.
            magnitude:            If >0, craft a one‐step adversarial example
                                  against each detector as in your original.
        """
        self.model = model
        self.loader = dataloader
        self.device = device
        self.suffix = suffix
        self.postprocessor_name = postprocessor_name
        
        self.cfg_dataset = cfg_dataset
        self.num_classes = cfg_dataset["num_classes"]
        
    
        self.scores = None  # Precomputed scores, if any
        self.latent_path = latent_path
        self.result_folder = result_folder

    def get_pertubated_scores(self, inputs, detector, magnitude):

        # inputs = inputs.to(self.device)
        inputs = inputs.to(self.device).detach().requires_grad_(True) 

        # inputs = Variable(inputs.clone(), requires_grad=True)
        adv_logits = self.model(inputs)
        scores = detector(logits=adv_logits)           # initial
        # backprop on log-score
        loss = torch.log(scores + 1e-12).sum()
        # loss = torch.log(scores.clamp_min(1e-12)).sum()
        # loss.backward()
        # step and detach
        grad_inputs, = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)
        with torch.no_grad():
            adv = inputs + magnitude * grad_inputs.sign()
        # inputs = (inputs + magnitude * inputs.grad.sign()).detach()
        # inputs = Variable(inputs, requires_grad=False)
        with torch.inference_mode():
            logits_adv = self.model(adv)
            scores_adv = detector(logits=logits_adv)
        # with torch.no_grad():
        #     scores_adv = detector(inputs=inputs)

        return scores_adv


    
    
    

    def get_scores(self, detectors, n_samples, list_configs, logits=None):

        n_det = len(detectors)
        if os.path.exists(self.latent_path):

            pkg = torch.load(self.latent_path, map_location="cpu")
            logits = pkg["logits"].to(torch.float32).to(self.device)        # (N, C)
            all_labels = pkg["labels"].numpy()                  # (N,)
            all_model_preds  = pkg["model_preds"].numpy()             # (N,)
            detector_labels_arr = (all_model_preds != all_labels)  # bool array

            if self.postprocessor_name in ["doctor", "odin", "relu"]:

                self.model.to(self.device)
                self.model.eval()
                all_scores = [np.zeros(n_samples, dtype=float) for _ in range(n_det)]
                for idx, dec in tqdm(enumerate(detectors), total=len(detectors), desc="Getting Detectors Scores", leave=False):
                    magnitude = list_configs[idx]['magnitude']
                    
                    if magnitude > 0:
                        
                        self.model.eval()
                        write=0
                        
                        for inputs, labels in self.loader:
                            bs = inputs.size(0)
                            scores = self.get_pertubated_scores(inputs, dec, magnitude)
                            if isinstance(scores, torch.Tensor):
                                scores = scores.detach().cpu().numpy()
                            all_scores[idx][write:write+bs] = scores
                            write += bs
                    else:
                        with torch.no_grad():
                            scores = dec(logits=logits).cpu().numpy()
                        all_scores[idx][:] = scores

            else:
                all_scores = [detector(logits=logits).cpu().numpy() for detector in detectors]
                # print("all scores shape:", np.array(all_scores).shape)
                # all_scores = [detector(logits=logits).cpu().numpy() for detector in detectors]

        else:
            
            all_scores = [np.zeros(n_samples, dtype=float) for _ in range(n_det)]
            all_labels = np.zeros(n_samples, dtype=int)
            all_model_preds = np.zeros(n_samples, dtype=int)
            detector_labels_arr = np.zeros(n_samples, dtype=bool)
            all_logits = np.zeros((n_samples, self.num_classes), dtype=float)
            

            # iterate once over data
            idx = 0

            self.model.to(self.device)
            self.model.eval()
            for inputs, labels in tqdm(self.loader, desc="Getting Detectors Scores", leave=False):
                bs = inputs.size(0)
                inputs = inputs.to(self.device)             
                labels = labels.to(self.device)

                # forward pass for model accuracy & labels
                with torch.no_grad():
                    logits = self.model(inputs)
                    model_preds = torch.argmax(logits, dim=1)


                det_lab = (model_preds != labels).cpu().numpy()
                detector_labels_arr[idx: idx+bs] = det_lab
                all_labels[idx: idx+bs] = labels.cpu().numpy()
                all_model_preds[idx: idx+bs] = model_preds.cpu().numpy()
                all_logits[idx: idx+bs] = logits.cpu().numpy()

                # now each detector
                for i, det in enumerate(detectors):
                    # -- optionally craft 1‑step adv example per detector
                    if self.postprocessor_name in ["doctor", "odin", "relu"]:
                        magnitude = list_configs[i]['magnitude']
                        if magnitude > 0:
                            scores = self.get_pertubated_scores(inputs, det, magnitude)
                        else:
                            with torch.no_grad():
                                scores = det(logits=logits)
                    else:
                        with torch.no_grad():
                            scores = det(logits=logits)
                    if isinstance(scores, torch.Tensor):
                        scores = scores.cpu().numpy()
                    all_scores[i][idx: idx+bs] = scores
                    
                idx += bs

            # AFTER (robust)
            parent = os.path.dirname(self.latent_path)
            os.makedirs(parent, exist_ok=True)

            if self.latent_path is not None:
                tmp = self.latent_path + ".tmp"
                torch.save(
                    {
                        "logits": torch.tensor(all_logits),     # compact on disk
                        "labels": torch.tensor(all_labels).to(torch.int64),
                        "model_preds": torch.tensor(all_model_preds).to(torch.int64),
                    },
                    tmp,
                )
                os.replace(tmp, self.latent_path)  # atomic rename
        # print("all_scores shape:", all_scores.shape)
        # print("detector_labels_arr shape:", detector_labels_arr.shape)
        self.scores = {
            "scores": all_scores,
            "detector_labels": detector_labels_arr
        } 

    def evaluate(
            self, 
            list_configs, 
            detectors=None, 
            all_scores=None, 
            detector_labels=None, 
            suffix=None) -> list[dict]:
        """
        Run the model once per batch, then each detector on those inputs.

        Args:
            detectors:  Either a list of detector‐objects or a dict name→detector.
                        Each detector must be callable as:
                            scores = detector(inputs=..., logits=...)
                        and return a 1‐D tensor of “outlier scores” (higher = more likely error).

        Returns:
            results: dict mapping detector_name → metrics dict, e.g.
                {
                  "ODIN": {
                      "fpr@95tpr": 0.12,
                      "roc_auc": 0.94,
                      "aupr_err": 0.88,
                      "aupr_in": 0.90,
                      "aurc": 0.23,
                      "model_acc": 0.79,  # same for all detectors
                  },
                  …
                }
        """
        # normalize detectors into an ordered dict name→detector

        # storage
        n_samples = len(self.loader.dataset)
        if all_scores is None:
            self.get_scores(detectors, n_samples, list_configs)
            all_scores = self.scores["scores"]
            detector_labels = self.scores["detector_labels"]  # bool array
     
        list_results = []
        for i, scores in enumerate(all_scores):
        
            results = compute_all_metrics(
                conf=scores,
                detector_labels=detector_labels,
                result_folder=self.result_folder,
            )
            
            results = pd.DataFrame([results])
            
            suffix = suffix if suffix is not None else self.suffix
            results.columns = [f"{col}_{suffix}" for col in results.columns]
            
            
            # config = _prepare_config_for_results(list_configs[i])
            # config = pd.json_normalize(config, sep="_")
            config = pd.DataFrame([list_configs[i]])
            results = pd.concat([config, results], axis=1)
            list_results.append(results)

        return list_results


class MultiDetectorEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        suffix = "train",
        base_config=None,

    ):
        """
        Evaluate a single model against multiple post-hoc detectors.

        Args:
            model:                A trained classifier (outputs raw logits).
            dataloader:           DataLoader for the evaluation set.
            device:               torch.device to run model & detectors on.
            magnitude:            If >0, craft a one‐step adversarial example
                                  against each detector as in your original.
        """
        self.model = model
        self.loader = dataloader
        self.device = device
        self.suffix = suffix
        self.base_config = base_config
        self.method_name = base_config['method_name']
        self.num_classes = base_config['data']['n_classes']
        self.scores = None  # Precomputed scores, if any
        self.root = f"storage_latent/{base_config['data']['name']}_{base_config['model']['name']}_{base_config['model']['preprocessor']}_r-{base_config['data']['ratio_calib']}_seed-split-{base_config['data']['seed_split']}/"
        if suffix == "train":
            if (self.method_name == "clustering") & (base_config['clustering']['space'] == "classifier"):

                self.latent_path = self.root + f"{base_config['clustering']['space']}_{suffix}_n-epochs{base_config['data']['n_epochs']}_transform-{base_config['data']['transform']}.pt"
            else:
                self.latent_path = self.root + f"logits_{suffix}_n-epochs{base_config['data']['n_epochs']}_transform-{base_config['data']['transform']}.pt"
        else:
            if (self.method_name == "clustering") & (base_config['clustering']['space'] == "classifier"):
                self.latent_path = self.root + f"{base_config['clustering']['space']}_{suffix}.pt"
            else:
                self.latent_path = self.root + f"logits_{suffix}.pt"

    def get_pertubated_scores(self, inputs, detector, magnitude):

        # inputs = inputs.to(self.device)
        inputs = inputs.to(self.device).detach().requires_grad_(True) 

        # inputs = Variable(inputs.clone(), requires_grad=True)
        adv_logits = self.model(inputs)
        scores = detector(logits=adv_logits)           # initial
        # backprop on log-score
        loss = torch.log(scores + 1e-12).sum()
        # loss = torch.log(scores.clamp_min(1e-12)).sum()
        # loss.backward()
        # step and detach
        grad_inputs, = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)
        with torch.no_grad():
            adv = inputs + magnitude * grad_inputs.sign()
        # inputs = (inputs + magnitude * inputs.grad.sign()).detach()
        # inputs = Variable(inputs, requires_grad=False)
        with torch.inference_mode():
            logits_adv = self.model(adv)
            scores_adv = detector(logits=logits_adv)
        # with torch.no_grad():
        #     scores_adv = detector(inputs=inputs)

        return scores_adv


    
    
    # def get_scores_without_pertubation(self, detectors, n_samples, list_configs, logits=None):
    #     n_det = len(detectors)

    #     if logits is None:
    #         if os.path.exists(self.latent_path):

    #             pkg = torch.load(self.latent_path, map_location="cpu")
    #             logits = pkg["logits"].to(torch.float32)        # (N, C)
    #             all_labels = pkg["labels"].numpy()                  # (N,)
    #             all_model_preds  = pkg["model_preds"].numpy()             # (N,)
    #             detector_labels_arr = (all_model_preds != all_labels)  # bool array

    #             all_scores = [detector(logits=logits).cpu().numpy() for detector in detectors]
    #         else:
    #             all_scores = [np.zeros(n_samples, dtype=float) for _ in range(n_det)]
    #             all_labels = np.zeros(n_samples, dtype=int)
    #             all_model_preds = np.zeros(n_samples, dtype=int)
    #             detector_labels_arr = np.zeros(n_samples, dtype=bool)
    #             all_logits = np.zeros((n_samples, self.num_classes), dtype=float)
                

    #             # iterate once over data
    #             idx = 0
            
    #             self.model.eval()
    #             for inputs, labels in tqdm(self.loader, desc="Getting Detectors Scores", leave=False):
    #                 bs = inputs.size(0)
    #                 inputs = inputs.to(self.device)             
    #                 labels = labels.to(self.device)

    #                 # forward pass for model accuracy & labels
    #                 with torch.no_grad():
    #                     logits = self.model(inputs)
    #                     model_preds = torch.argmax(logits, dim=1)



    #                 det_lab = (model_preds != labels).cpu().numpy()
    #                 detector_labels_arr[idx: idx+bs] = det_lab
    #                 all_labels[idx: idx+bs] = labels.cpu().numpy()
    #                 all_model_preds[idx: idx+bs] = model_preds.cpu().numpy()
    #                 all_logits[idx: idx+bs] = logits.cpu().numpy()
                        
    #                 idx += bs
    #             all_scores = [detector(
    #                 logits=torch.tensor(all_logits).to(detector.device)).cpu().numpy() for detector in detectors]


    

    def get_scores(self, detectors, n_samples, list_configs, logits=None):

        n_det = len(detectors)
        if os.path.exists(self.latent_path):

            pkg = torch.load(self.latent_path, map_location="cpu")
            logits = pkg["logits"].to(torch.float32).to(self.device)        # (N, C)
            all_labels = pkg["labels"].numpy()                  # (N,)
            all_model_preds  = pkg["model_preds"].numpy()             # (N,)
            detector_labels_arr = (all_model_preds != all_labels)  # bool array

            if self.method_name in ["gini", "max_proba", "metric_learning"]:

                self.model.to(self.device)
                self.model.eval()
                all_scores = [np.zeros(n_samples, dtype=float) for _ in range(n_det)]
                for idx, dec in tqdm(enumerate(detectors), total=len(detectors), desc="Getting Detectors Scores", leave=False):
                    magnitude = list_configs[idx][self.method_name]['magnitude']
                    
                    if magnitude > 0:
                        
                        self.model.eval()
                        write=0
                        
                        for inputs, labels in self.loader:
                            bs = inputs.size(0)
                            scores = self.get_pertubated_scores(inputs, dec, magnitude)
                            if isinstance(scores, torch.Tensor):
                                scores = scores.detach().cpu().numpy()
                            all_scores[idx][write:write+bs] = scores
                            write += bs
                    else:
                        with torch.no_grad():
                            scores = dec(logits=logits).cpu().numpy()
                        all_scores[idx][:] = scores

            else:
                all_scores = [detector(logits=logits).cpu().numpy() for detector in detectors]
                # print("all scores shape:", np.array(all_scores).shape)
                # all_scores = [detector(logits=logits).cpu().numpy() for detector in detectors]

        else:
            
            all_scores = [np.zeros(n_samples, dtype=float) for _ in range(n_det)]
            all_labels = np.zeros(n_samples, dtype=int)
            all_model_preds = np.zeros(n_samples, dtype=int)
            detector_labels_arr = np.zeros(n_samples, dtype=bool)
            all_logits = np.zeros((n_samples, self.num_classes), dtype=float)
            

            # iterate once over data
            idx = 0

            self.model.to(self.device)
            self.model.eval()
            for inputs, labels in tqdm(self.loader, desc="Getting Detectors Scores", leave=False):
                bs = inputs.size(0)
                inputs = inputs.to(self.device)             
                labels = labels.to(self.device)

                # forward pass for model accuracy & labels
                with torch.no_grad():
                    logits = self.model(inputs)
                    model_preds = torch.argmax(logits, dim=1)


                det_lab = (model_preds != labels).cpu().numpy()
                detector_labels_arr[idx: idx+bs] = det_lab
                all_labels[idx: idx+bs] = labels.cpu().numpy()
                all_model_preds[idx: idx+bs] = model_preds.cpu().numpy()
                all_logits[idx: idx+bs] = logits.cpu().numpy()

                # now each detector
                for i, det in enumerate(detectors):
                    # -- optionally craft 1‑step adv example per detector
                    if self.method_name in ["gini", "max_proba", "metric_learning"]:
                        magnitude = list_configs[i][self.method_name]['magnitude']
                        if magnitude > 0:
                            scores = self.get_pertubated_scores(inputs, det, magnitude)
                        else:
                            with torch.no_grad():
                                scores = det(logits=logits)
                    else:
                        with torch.no_grad():
                            scores = det(logits=logits)
                    if isinstance(scores, torch.Tensor):
                        scores = scores.cpu().numpy()
                    all_scores[i][idx: idx+bs] = scores
                    
                idx += bs

            # AFTER (robust)
            parent = os.path.dirname(self.latent_path)
            os.makedirs(parent, exist_ok=True)

            if self.latent_path is not None:
                tmp = self.latent_path + ".tmp"
                torch.save(
                    {
                        "logits": torch.tensor(all_logits),     # compact on disk
                        "labels": torch.tensor(all_labels).to(torch.int64),
                        "model_preds": torch.tensor(all_model_preds).to(torch.int64),
                    },
                    tmp,
                )
                os.replace(tmp, self.latent_path)  # atomic rename
        # print("all_scores shape:", all_scores.shape)
        # print("detector_labels_arr shape:", detector_labels_arr.shape)
        self.scores = {
            "scores": all_scores,
            "detector_labels": detector_labels_arr
        } 

    def evaluate(
            self, 
            list_configs, 
            detectors=None, 
            all_scores=None, 
            detector_labels=None, 
            suffix=None) -> list[dict]:
        """
        Run the model once per batch, then each detector on those inputs.

        Args:
            detectors:  Either a list of detector‐objects or a dict name→detector.
                        Each detector must be callable as:
                            scores = detector(inputs=..., logits=...)
                        and return a 1‐D tensor of “outlier scores” (higher = more likely error).

        Returns:
            results: dict mapping detector_name → metrics dict, e.g.
                {
                  "ODIN": {
                      "fpr@95tpr": 0.12,
                      "roc_auc": 0.94,
                      "aupr_err": 0.88,
                      "aupr_in": 0.90,
                      "aurc": 0.23,
                      "model_acc": 0.79,  # same for all detectors
                  },
                  …
                }
        """
        # normalize detectors into an ordered dict name→detector

        # storage
        n_samples = len(self.loader.dataset)
        if all_scores is None:
            self.get_scores(detectors, n_samples, list_configs)
            all_scores = self.scores["scores"]
            detector_labels = self.scores["detector_labels"]  # bool array
     
        list_results = []
        for i, scores in enumerate(all_scores):
        
            fpr, tpr, thr, auroc, accuracy, aurc_value, aupr_err, aupr_success = compute_all_metrics(
                conf=scores,
                detector_labels=detector_labels,
            )
        
            results = pd.DataFrame([{
                "fpr": fpr,
                "tpr": tpr,
                "thr": thr,
                "roc_auc": auroc,
                "model_acc": accuracy,
                "aurc": aurc_value,
                "aupr_err": aupr_err,
                "aupr_success": aupr_success,
            }])
            
            suffix = suffix if suffix is not None else self.suffix
            results.columns = [f"{col}_{suffix}" for col in results.columns]
            
            config = _prepare_config_for_results(list_configs[i])
            config = pd.json_normalize(config, sep="_")
            results = pd.concat([config, results], axis=1)
            list_results.append(results)

        return list_results


import os

# class DetectorEvaluator:
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         dataloader: torch.utils.data.DataLoader,
#         device: torch.device,
#         magnitude=0,
#         suffix = "train",
#         scores = None,
#         base_config=None,
#         seed_split=None,
#     ):
#         """
#         Evaluate a single model against multiple post-hoc detectors.

#         Args:
#             model:                A trained classifier (outputs raw logits).
#             dataloader:           DataLoader for the evaluation set.
#             device:               torch.device to run model & detectors on.
#             magnitude:            If >0, craft a one‐step adversarial example
#                                   against each detector as in your original.
#         """
#         self.model = model.to(device)
#         self.loader = dataloader
#         self.device = device
#         self.magnitude = magnitude
#         self.suffix = suffix
#         self.scores = None  # Precomputed scores, if an
#         self.base_config = base_config
#         self.num_classes = base_config['data']['n_classes']
#         self.seed_split = seed_split
#         self.latent_path = f"storage_latent/{base_config['data']['name']}_{base_config['model']['name']}_r-{base_config['data']['r']}_seed-split-{seed_split}/logits_{suffix}.pt"


#     def get_scores(self, detector, n_samples):

#         if os.path.exists(self.latent_path):
#             pkg = torch.load(self.latent_path, map_location="cpu")
#             logits = pkg["logits"].to(torch.float32)        # (N, C)
#             all_labels = pkg["labels"].numpy()                  # (N,)
#             all_model_preds  = pkg["model_preds"].numpy()             # (N,)
#             detector_labels_arr = (all_model_preds != all_labels)  # bool array
#             all_scores = detector(logits=logits).cpu().numpy()
#             # print("all_scores shape:", all_scores.shape)
#             # print("detector_labels_arr shape:", detector_labels_arr.shape)
#         else:

#             all_scores = np.zeros(n_samples, dtype=float)
#             # detector_labels = model_pred != true_label  (same for all detectors)
#             all_labels = np.zeros(n_samples, dtype=int)
#             all_model_preds = np.zeros(n_samples, dtype=int)
#             all_logits = np.zeros((n_samples, self.num_classes), dtype=float)
#             detector_labels_arr = np.zeros(n_samples, dtype=bool)

#             # iterate once over data
#             idx = 0
#             self.model.eval()
#             for inputs, labels in tqdm(self.loader, desc="Getting logits", leave=False):
#                 bs = inputs.size(0)
#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)

#                 # forward pass for model accuracy & labels
#                 with torch.no_grad():
#                     logits = self.model(inputs)
#                     model_preds = torch.argmax(logits, dim=1)

#                 # which examples are errors?
#                 det_lab = (model_preds != labels).cpu().numpy()
#                 detector_labels_arr[idx: idx+bs] = det_lab

#                 # store global labels & preds
#                 all_logits[idx: idx+bs] = logits.cpu().numpy()
#                 all_labels[idx: idx+bs] = labels.cpu().numpy()
#                 all_model_preds[idx: idx+bs] = model_preds.cpu().numpy()

#                 # now each detector

#                 if self.magnitude > 0:
#                     # clone input for gradient
#                     adv_in = Variable(inputs.clone(), requires_grad=True)
#                     adv_logits = self.model(adv_in)
#                     scores0 = detector(logits=adv_logits)           # initial
#                     # backprop on log-score
#                     loss = torch.log(scores0 + 1e-12).sum()
#                     loss.backward()
#                     # step and detach
#                     adv_in = (adv_in - self.magnitude[i] * adv_in.grad.sign()).detach()
#                     with torch.no_grad():
#                         scores = detector(inputs=adv_in)
#                 else:
#                     with torch.no_grad():
#                         scores = detector(logits=logits)
#                 if isinstance(scores, torch.Tensor):
#                     scores = scores.cpu().numpy()
#                 all_scores[idx: idx+bs] = scores

#                 idx += bs


#             # AFTER (robust)        
#             # AFTER (robust)
#             parent = os.path.dirname(self.latent_path)
#             os.makedirs(parent, exist_ok=True)

#             tmp = self.latent_path + ".tmp"
#             torch.save(
#                 {
#                     "logits": torch.tensor(all_logits),     # compact on disk
#                     "labels": torch.tensor(all_labels).to(torch.int64),
#                     "model_preds": torch.tensor(all_model_preds).to(torch.int64),
#                 },
#                 tmp,
#             )
#             os.replace(tmp, self.latent_path)  # atomic rename
#         # print("all_scores shape:", all_scores.shape)
#         # print("detector_labels_arr shape:", detector_labels_arr.shape)
#         self.scores = {
#             "scores": all_scores,
#             "labels": all_labels,
#             "model_preds": all_model_preds,
#             "detector_labels": detector_labels_arr
#         } 

#     def evaluate(self, detector):
#         """
#         Run the model once per batch, then each detector on those inputs.

#         Args:
#             detectors:  Either a list of detector‐objects or a dict name→detector.
#                         Each detector must be callable as:
#                             scores = detector(inputs=..., logits=...)
#                         and return a 1‐D tensor of “outlier scores” (higher = more likely error).

#         Returns:
#             results: dict mapping detector_name → metrics dict, e.g.
#                 {
#                   "ODIN": {
#                       "fpr@95tpr": 0.12,
#                       "roc_auc": 0.94,
#                       "aupr_err": 0.88,
#                       "aupr_in": 0.90,
#                       "aurc": 0.23,
#                       "model_acc": 0.79,  # same for all detectors
#                   },
#                   …
#                 }
#         """
#         # normalize detectors into an ordered dict name→detector

#         # storage

#         n_samples = len(self.loader.dataset)
#         if self.scores is None:
#             self.get_scores(detector, n_samples)

#         all_scores = self.scores["scores"]
#         detector_labels_arr = self.scores["detector_labels"]  # bool array
    
#         fpr, tpr, thr, auroc, accuracy, aurc_value, aupr_err, aupr_success = compute_all_metrics(
#                         conf=all_scores,
#                         detector_labels=detector_labels_arr,
#                     )

#         results = pd.DataFrame([{
#                 "fpr": fpr,
#                 "tpr": tpr,
#                 "thr": thr,
#                 "roc_auc": auroc,
#                 "model_acc": accuracy,
#                 "aurc": aurc_value,
#                 "aupr_err": aupr_err,
#                 "aupr_success": aupr_success,
#         }])
            
#         results.columns = [f"{col}_{self.suffix}" for col in results.columns]
    

#         return results