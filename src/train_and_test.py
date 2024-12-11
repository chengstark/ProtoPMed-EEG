import time
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
import numpy as np
import pandas as pd
import csv

DEBUG_PRINT = False

def debug_print(msg, flush=True):
    if DEBUG_PRINT:
        print(msg, flush=flush)

def _train_or_test(model, dataloader, optimizer=None, use_l1_mask=True,
                   coefs=None, log=print, save_logits=False, epoch=20):
    """
    Train or test the model on the provided dataset.

    Args:
        model: PyTorch model (supports multi-GPU).
        dataloader: DataLoader for the dataset.
        optimizer: Optimizer for training; None for testing.
        use_l1_mask: Whether to use L1 mask on the last layer weights.
        coefs: Dictionary of coefficients for various loss components.
        log: Logging function.
        save_logits: Whether to save logits during testing.
        epoch: Current epoch number.
    Returns:
        Tuple[float, float]: Accuracy and seizure ROC AUC score.
    """
    is_train = optimizer is not None
    start = time.time()
    n_examples, n_correct, n_batches = 0, 0, 0
    total_cross_entropy = total_cluster_cost = 0
    total_ortho_loss = total_separation_cost = total_avg_separation_cost = total_fa_cost = 0

    total_output, total_one_hot_label = [], []
    all_targets, all_predictions = [], []

    for batch_idx, (image, label, patient_id) in enumerate(dataloader):
        batch_start_time = time.time()
        input = image.cuda()
        target = label.cuda()

        with (torch.enable_grad() if is_train else torch.no_grad()):
            prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).cuda()
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class

            logits, marginless_logits, max_activations = model(input, prototypes_of_wrong_class=None)

            class_weight = torch.tensor([0.783, 12.129, 0.838, 0.608, 0.992, 1.257], dtype=torch.float32).cuda()
            cross_entropy = torch.nn.functional.cross_entropy(logits, target, weight=class_weight)
            orthogonality_loss = model.module.get_prototype_orthogonalities()
            total_ortho_loss += orthogonality_loss.item()

            if not is_train and save_logits:
                _output_scores = [','.join(map(str, scores.cpu().numpy())) for scores in logits]
                write_file = './logit_csvs/logits.csv'
                with open(write_file, 'a') as logit_file:
                    writer = csv.writer(logit_file)
                    for _index, patient in enumerate(patient_id):
                        writer.writerow([patient, _output_scores[_index]])
                log(f'Wrote logits to {write_file}.')

            correct_class_prototype_activations, _ = torch.max(max_activations * prototypes_of_correct_class, dim=1)
            cluster_cost = -torch.mean(correct_class_prototype_activations)

            incorrect_class_prototype_activations, _ = torch.max(max_activations * prototypes_of_wrong_class, dim=1)
            separation_cost = -torch.mean(incorrect_class_prototype_activations)

            avg_separation_cost = torch.mean(
                torch.sum(max_activations * prototypes_of_wrong_class, dim=1) /
                torch.sum(prototypes_of_wrong_class, dim=1)
            )

            if use_l1_mask:
                l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
            else:
                l1 = model.module.last_layer.weight.norm(p=1)

            _, predicted = torch.max(marginless_logits.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            all_predictions.append(predicted.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            one_hot_label = np.zeros((len(target), model.module.num_classes))
            for k, tgt in enumerate(target):
                one_hot_label[k][tgt.item()] = 1

            prob = torch.nn.functional.softmax(logits, dim=1)
            total_output.extend(prob.cpu().numpy())
            total_one_hot_label.extend(one_hot_label)

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        if is_train:
            loss = (coefs['crs_ent'] * cross_entropy +
                    coefs['clst'] * cluster_cost +
                    coefs['sep'] * separation_cost +
                    coefs['l1'] * l1 +
                    coefs['ortho'] * orthogonality_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    end = time.time()
    log(f'Total time: {end - start:.2f}s')
    log(f'Cross entropy: {total_cross_entropy / n_batches:.4f}')
    log(f'Cluster cost: {total_cluster_cost / n_batches:.4f}')
    log(f'Orthogonality loss: {total_ortho_loss / n_batches:.4f}')
    log(f'Separation cost: {total_separation_cost / n_batches:.4f}')
    log(f'Average separation cost: {total_avg_separation_cost / n_batches:.4f}')

    cm = confusion_matrix(
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        normalize='true'
    )
    log(f'Confusion matrix:\n{cm}')

    avg_roc_auc, avg_pr_auc, seizure_roc_auc = 0, 0, None
    for class_idx in range(len(total_one_hot_label[0])):
        class_roc_auc = roc_auc_score(
            np.array(total_one_hot_label)[:, class_idx],
            np.array(total_output)[:, class_idx]
        )
        avg_roc_auc += class_roc_auc / len(total_one_hot_label[0])
        log(f'ROC AUC for class {class_idx}: {class_roc_auc:.4f}')

        if class_idx == 1:
            seizure_roc_auc = class_roc_auc

        precision, recall, _ = precision_recall_curve(
            np.array(total_one_hot_label)[:, class_idx],
            np.array(total_output)[:, class_idx]
        )
        pr_auc = auc(recall, precision)
        avg_pr_auc += pr_auc / len(total_one_hot_label[0])
        log(f'PR AUC for class {class_idx}: {pr_auc:.4f}')

    log(f'Accuracy: {n_correct / n_examples * 100:.2f}%')
    return n_correct / n_examples, seizure_roc_auc

def train(model, dataloader, optimizer, coefs=None, log=print, epoch=20):
    """Train the model on the dataset."""
    assert optimizer is not None
    log('Starting training')
    model.train()
    return _train_or_test(model, dataloader, optimizer, coefs, log, epoch=epoch)

def test(model, dataloader, log=print, save_logits=False, epoch=20):
    """Test the model on the dataset."""
    log('Starting testing')
    model.eval()
    return _train_or_test(model, dataloader, optimizer=None, log=log, save_logits=save_logits, epoch=epoch)

def last_only(model, log=print):
    """Freeze all layers except the last layer."""
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    log('Last layer training only')

def warm_only(model, log=print):
    """Unfreeze add-on layers and prototype vectors for training."""
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    log('Warm training enabled')

def joint(model, log=print):
    """Unfreeze all layers for joint training."""
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    log('Joint training enabled')
