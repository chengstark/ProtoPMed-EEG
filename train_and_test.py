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
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None 
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_ortho_loss = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_fa_cost = 0

    total_output = []
    total_one_hot_label = []

    all_targets = []
    all_predictions = []

    debug_print(f'F1 time {time.time() - start} at {time.time()}', flush=True)


    for batch_idx, (image, label, patient_id) in enumerate(dataloader):
        
        debug_print(f'Exec batch {batch_idx} time {time.time() - start} at {time.time()}', flush=True)

        batch_start_time = time.time()

        input = image.cuda()
        target = label.cuda()
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()

        debug_print(f'\tF2 {time.time() - batch_start_time} at {time.time()}', flush=True)

        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class

            logits, marginless_logits, max_activations = model(input, prototypes_of_wrong_class=None)
            
            debug_print(f'\tF-aft forward {time.time() - batch_start_time} at {time.time()}', flush=True)


            class_weight = torch.from_numpy(np.asarray([0.78344142, 12.1291834,   0.838353,    0.60804644,  0.99168172,  1.25735498])).type(torch.FloatTensor).cuda()

            cross_entropy = torch.nn.functional.cross_entropy(logits, target, weight=class_weight)
            # orthogonalities = model.module.get_prototype_orthogonalities_old()
            # orthogonality_loss = torch.norm(orthogonalities)
            orthogonality_loss = model.module.get_prototype_orthogonalities()
            total_ortho_loss += orthogonality_loss.item()

            debug_print(f'\tF-aft loss {time.time() - batch_start_time} at {time.time()}', flush=True)

            # only save to csv on test
            if not is_train and save_logits:
                _output_scores = [",".join([str(score) for score in scores.cpu().numpy()]) for scores in logits]
                write_file = './logit_csvs/0709_trainplusval_3_class_margin_logits.csv'
                with open(write_file, mode='a') as logit_file:
                    logit_writer = csv.writer(logit_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for _index in range(len(patient_id)):
                        logit_writer.writerow([patient_id[_index], _output_scores[_index]])
                log(f'Wrote to {write_file}.')

                debug_print(f'\tF-aft save logits {time.time() - batch_start_time} at {time.time()}', flush=True)


            # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
            # calculate cluster cost
            correct_class_prototype_activations, _ = torch.max(max_activations * prototypes_of_correct_class, dim=1)
            cluster_cost = -torch.mean(correct_class_prototype_activations)

            debug_print(f'\tF-aft cluster cost {time.time() - batch_start_time} at {time.time()}', flush=True)


            # calculate separation cost
            incorrect_class_prototype_activations, _ =  torch.max(max_activations * prototypes_of_wrong_class, dim=1)
            separation_cost = -torch.mean(incorrect_class_prototype_activations)

            debug_print(f'\tF-aft separation cost {time.time() - batch_start_time} at {time.time()}', flush=True)

            # calculate avg seperation cost
            avg_separation_cost = \
                torch.sum(max_activations * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)

            debug_print(f'\tF-aft avg separation cost {time.time() - batch_start_time} at {time.time()}', flush=True)

            if use_l1_mask:
                l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
            else:
                l1 = model.module.last_layer.weight.norm(p=1) 
            
            # evaluation statistics
            _, predicted = torch.max(marginless_logits.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            debug_print(f'\tF-aft predictions {time.time() - batch_start_time} at {time.time()}', flush=True)

            # confusion matrix
            all_predictions.append(predicted.squeeze().cpu().numpy())
            all_targets.append(target.squeeze().cpu().numpy())

            one_hot_label = np.zeros(shape=(len(target), model.module.num_classes))
            for k in range(len(target)):
                one_hot_label[k][target[k].item()] = 1

            prob = torch.nn.functional.softmax(logits, dim=1)
            total_output.extend(prob.data.cpu().numpy())
            total_one_hot_label.extend(one_hot_label)

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

            debug_print(f'\tF-before backward {time.time() - batch_start_time} at {time.time()}', flush=True)


        # compute gradient and do SGD step
        if is_train:
            loss = (coefs['crs_ent'] * cross_entropy
                    + coefs['clst'] * cluster_cost
                    + coefs['sep'] * separation_cost
                    + coefs['l1'] * l1
                    + coefs['ortho'] * orthogonality_loss)
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            debug_print(f'\tF-aft backward {time.time() - batch_start_time} at {time.time()}', flush=True)


        del input
        del target
        del logits
        del predicted
        del max_activations
        del marginless_logits

        debug_print(f'\tF-end {time.time() - batch_start_time} at {time.time()}', flush=True)


    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    log('\torthogonality loss:\t{0}'.format(total_ortho_loss / n_batches))
    log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    log('\tfine annotation:\t{0}'.format(total_fa_cost / n_batches))
    log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))

    cm = confusion_matrix(np.concatenate(all_targets, axis=0), np.concatenate(all_predictions, axis=0), normalize='true')

    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    if epoch < 10 or epoch % 10 == 0:
        log('\tthe confusion matrix is: \n{0}'.format(cm))

    avg_roc_auc = 0
    avg_pr_auc = 0
    seizure_roc_auc = None
    for class_idx in range(len(total_one_hot_label[0])):
        class_roc_auc = roc_auc_score(np.array(total_one_hot_label)[:, class_idx], np.array(total_output)[:, class_idx])
        avg_roc_auc += class_roc_auc / len(total_one_hot_label[0])
        log("\troc auc score for class {} is: \t\t{}".format(class_idx, class_roc_auc))

        if class_idx == 1:
            seizure_roc_auc = class_roc_auc
        
        class_precision, class_recall, _ = precision_recall_curve(np.array(total_one_hot_label)[:, class_idx], np.array(total_output)[:, class_idx])
        class_pr_auc = auc(class_recall, class_precision)
        avg_pr_auc += class_pr_auc / len(total_one_hot_label[0])
        log("\tpr auc score for class {} is: \t\t{}".format(class_idx, class_pr_auc))

    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    
    return n_correct / n_examples, seizure_roc_auc


def train(model, dataloader, optimizer, coefs=None, log=print, epoch=20):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer, coefs=coefs, log=log, epoch=epoch)


def test(model, dataloader, log=print, save_logits=False, epoch=20):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None, log=log, save_logits=save_logits, epoch=epoch)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
