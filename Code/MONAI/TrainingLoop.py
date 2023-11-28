import numpy as np
from monai.transforms import (
    AsDiscrete,
    Compose,
)
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
import torch, os
from datetime import datetime

root_dir = r'C:\Users\Nicolai\PycharmProjects\ML4MedWS2023'
current_datetime = datetime.now()
formatted_date = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")


def TRAINING(model,
             NUM_CLASSES: int,
             MAX_EPOCHS: int,
             VAL_INTERVAL: int,
             train_loader,
             val_loader,
             optimizer,
             loss_function,
             metrics,
             # metric_name: str,
             train_ds,
             device = torch.device("cpu:0"),
             ):
    # max_epochs = 600
    # val_interval = 1
    best_metric = {}
    for name, metric in metrics.items():  # ToDo: choose what is best depending on metric (dice higher, hausdorff lower). implement this dependence. at the moment hardcoded
        if 'hausdorff' in name.lower():
            best_metric[name] = np.Inf
        elif 'dice' in name.lower():
            best_metric[name] = -1
    # best_metric = {name: -1 for name, metric in metrics.items()}
    best_metric_epoch = {name: -1 for name, metric in metrics.items()}
    metric_values = {name: [] for name, metric in metrics.items()}
    epoch_loss_values = []

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
    post_label = Compose([AsDiscrete(to_onehot=NUM_CLASSES)])

    for epoch in range(MAX_EPOCHS):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{MAX_EPOCHS}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    # metric(y_pred=val_outputs, y=val_labels)
                    # Evaluate multiple metrics
                    for name, metric in metrics.items():
                        metric(y_pred=val_outputs, y=val_labels)
                    # metric_results = {name: metric(y_pred=val_outputs, y=val_labels) for name, metric in metrics.items()}
                    metric_results = {name: metric.aggregate().item() for name, metric in metrics.items()}

                # aggregate the final mean dice result
                # metric = metric.aggregate().item()
                # reset the status for next validation round
                # metric.reset()
                def HelperBestMetric(name, metric, best_metric, best_metric_epoch):
                    best_metric[name] = metric
                    best_metric_epoch[name] = epoch + 1

                    torch.save(model.state_dict(), os.path.join(root_dir, 'Model', 'Save',
                                                                f"best_metric_{name}_model_{formatted_date}.pth"))
                    print(f"saved new best metric {name} model")
                    return best_metric, best_metric_epoch

                for name, metric in metric_results.items():  # ToDo: choose what is best depending on metric (dice higher, hausdorff lower). implement this dependence. at the moment hardcoded
                    metric_values[name].append(metric)
                    if 'hausdorff' in name.lower():
                        if metric < best_metric[name]:
                            best_metric, best_metric_epoch = HelperBestMetric(name, metric, best_metric, best_metric_epoch)
                    elif 'dice' in name.lower():
                        if metric > best_metric[name]:
                            best_metric, best_metric_epoch = HelperBestMetric(name, metric, best_metric, best_metric_epoch)
                    metrics[name].reset()

                    print(
                        # f"current epoch: {epoch + 1} current mean {metric_results}: {metric:.4f}"
                        f"current epoch: {epoch + 1} current mean {name}: {metric:.4f}"
                        f"\nbest mean: {best_metric[name]:.4f} "
                        f"at epoch: {best_metric_epoch[name]}"
                    )
        #%%
    for name, metric in metric_results.items():
        print(f"train completed, best_metric {name}: {best_metric[name]:.4f} " f"at epoch: {best_metric_epoch[name]}")
    return model