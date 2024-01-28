import numpy as np
import torch, os

from monai.transforms import (
    AsDiscrete,
    Compose,
)
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

from datetime import datetime
from tqdm import tqdm

from Code.MONAI import SlidingWindowConfig

if os.path.isdir(r'.\ML4MedWS2023'):
    root_dir = r'.\ML4MedWS2023'
else:
    root_dir = os.getcwd()

current_datetime = datetime.now()
formatted_date = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")


def TRAINING_withoutSlidingWindow_withScheduler(model,
                                                NUM_CLASSES: int,
                                                MAX_EPOCHS: int,
                                                VAL_INTERVAL: int,
                                                train_loader,
                                                val_loader,
                                                optimizer,
                                                scheduler,
                                                loss_function,
                                                metricCalculator,
                                                train_ds,
                                                indicator: str,
                                                device=torch.device("cpu:0"),
                                                ):
    # max_epochs = 600
    # val_interval = 1

    best_metric = -1
    best_metric_epoch = -1
    metric_values = []
    epoch_loss_values = []
    metric_per_class_values = []

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
    post_label = Compose([AsDiscrete(to_onehot=NUM_CLASSES)])

    for epoch in range(MAX_EPOCHS):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{MAX_EPOCHS} on device {device}")
        model.train()
        epoch_loss = 0
        step = 0
        # for batch_data in tqdm(train_loader):
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
        scheduler.step(epoch_loss)

        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f} LR: {optimizer.param_groups[0]['lr']}")

        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                # for val_data in tqdm(val_loader):
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    # val_outputs = sliding_window_inference(val_inputs, SlidingWindowConfig.roi_size, SlidingWindowConfig.batch_size, model)

                    val_outputs = model(val_inputs)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                    # Evaluate dice metrics
                    metricCalculator(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = metricCalculator.aggregate().item()
                # reset the status for next validation round
                metric_per_class = metric.aggregate(reduction="none").mean(axis=0)
                metricCalculator.reset()
                # Save the metric value
                metric_values.append(metric)
                metric_per_class_values.append(metric_per_class)
                # Check if current model is better than the best one
                if metric > best_metric:
                    # Update the best metric that is determined so far
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, f"best_metric_dice_model_{str(epoch)}_{formatted_date}_{indicator}.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    return model, epoch_loss_values, metric_values
