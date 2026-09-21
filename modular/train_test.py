from tqdm.auto import tqdm
from timeit import default_timer as timer
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb
from modular.utility import (
    log_confusion_matrix,
    log_f1_by_class,
    log_recall_by_class,
    log_confusion_matrix_with_distribution,
)

def train_step(model: torch.nn.Module, 
               train_dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               metrics,
               device: torch.device,
              ):
    train_loss=0
    # put model into training mode
    model.train()

    # add a loop to loop through the training batches
    for batch, (X,y) in enumerate (train_dataloader):
        # Put data on target device
        X=X.to(device)
        y=y.to(device)

        # 1. Forward pass
        y_logits=model(X)   # result of model
        y_pred_probs = torch.log_softmax(y_logits, dim=1) # dim=1 calculates log-probabilities across rows. # Use logsoftmax function on our model logits to turn them into prediction probabilities. the rows of LogSoftmax outputs will not add up to 1. Instead, if you take the exponent (\(e^{x}\)) of each number in a row, those exponents will add up 1
        y_preds=torch.argmax(y_pred_probs, dim=1)  # find the predicted labels

        #2. Calculate Loss
        loss =loss_fn(y_logits, y)
        train_loss+= loss.item()  # accumilate train loass
        metrics.update(y_preds, y)

        #3. optimizer. Have to zero it for each epoch
        optimizer.zero_grad()

        #4. Back propagration
        loss.backward()

        #5. Step the optimizer
        optimizer.step()


    # divide total train loss and acc by length of train datest_dataloader
    train_loss /=len(train_dataloader) # if u want you can use something like train_loss_avg = train_loss / len(train_dataloader)

    # Calculate metrics for entire epoch
    train_metric_results = metrics.compute()

    # Reset for next epoch
    metrics.reset()


    return train_loss, train_metric_results

def test_step(model: torch.nn.Module, 
               test_dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               metrics,
               device: torch.device,
              ):
    test_loss=0 
    model.eval() # turns off different settings in the model not needed for evaluation

    with torch.inference_mode(): # turns off graident tracking and a couple of other things
        for X,y in test_dataloader:
            X=X.to(device)
            y=y.to(device)

            #1. Do forward pass
            test_logits=model(X)   #
            test_probs = torch.log_softmax(test_logits, dim=1) 
            test_pred=torch.argmax(test_probs, dim=1) 

            #2. Calculate the loss
            test_loss += loss_fn(test_logits, y).item()
            #3. Calculate the acc
            metrics.update(test_pred, y)

        # calculate the test loss average per test
        test_loss/=len(test_dataloader)

        # Calculate metrics over entire validation dataset
        test_metric_results = metrics.compute()

        # Reset before next epoch
        metrics.reset()      

    return test_loss, test_metric_results


# 1. Take in various parameters required for training and test steps
def train_test(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        train_metrics,
        test_metrics,
        device,
        class_names,
        loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
        epochs: int = 5
):
    
    # 2. Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "train_precision": [],
        "train_recall": [],
        
        "test_loss": [],
        "test_acc": [],
        "test_f1": [],
        "test_precision": [],
        "test_recall": [],

        "epoch_time": []
}
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        
        epoch_start = timer()
        
        train_loss, train_results = train_step(model=model,
                                           train_dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           metrics=train_metrics,
                                            device=device)
                                           
        test_loss, test_results = test_step(model=model,
                                        test_dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        metrics=test_metrics,
                                        device=device)

        epoch_end = timer()
        epoch_time = epoch_end - epoch_start
        
        # 4. Print epoch results
        print(
            f"Epoch: {epoch+1} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_results['train_accuracy'].item()*100:.2f}% | "
            f"Train F1: {train_results['train_f1'].item():.4f} | "
            f"train_precision: {train_results['train_precision'].item():.4f} | "
            f"train_recall: {train_results['train_recall'].item():.4f} | "

            f"Test loss: {test_loss:.4f} | "
            f"Test acc: {test_results['test_accuracy'].item()*100:.2f}% | "
            f"Test F1: {test_results['test_f1'].item():.4f} | "
            f"test_precision: {test_results['test_precision'].item():.4f} | "
            f"test_recall: {test_results['test_recall'].item():.4f}"
        )

        # 4. Save training results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_results["train_accuracy"].item())
        results["train_f1"].append(train_results["train_f1"].item())
        results["train_precision"].append(train_results["train_precision"].item())
        results["train_recall"].append(train_results["train_recall"].item())

        # 5. Save test results
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_results["test_accuracy"].item())
        results["test_f1"].append(test_results["test_f1"].item())
        results["test_precision"].append(test_results["test_precision"].item())
        results["test_recall"].append(test_results["test_recall"].item())

        results["epoch_time"].append(epoch_time)
        
        # 6. Log to wandb
        wandb.log({
            "epoch": epoch + 1,
        
            "train/loss": train_loss,
            "train/accuracy": train_results["train_accuracy"].item(),
            "train/f1": train_results["train_f1"].item(),
            "train/precision": train_results["train_precision"].item(),
            "train/recall": train_results["train_recall"].item(),
        
            "val/loss": test_loss,
            "val/accuracy": test_results["test_accuracy"].item(),
            "val/f1": test_results["test_f1"].item(),
            "val/precision": test_results["test_precision"].item(),
            "val/recall": test_results["test_recall"].item(),
        
            "epoch_time": epoch_time
        })
    return results

def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    metrics,
    class_names,
    device: torch.device
):
    model.eval()
    metrics.reset()

    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Evaluating metrics", unit="batch"):
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            metrics.update(preds, y)

    metric_results = metrics.compute()
    metrics.reset()



    return metric_results