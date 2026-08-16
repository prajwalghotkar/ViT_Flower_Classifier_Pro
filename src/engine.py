import torch
from tqdm.auto import tqdm


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss, total_acc = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        total_acc += (pred_class == y).sum().item() / len(y_pred)

    return total_loss / len(dataloader), total_acc / len(dataloader)


def test_step(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()

            pred_class = y_pred.argmax(dim=1)
            total_acc += (pred_class == y).sum().item() / len(pred_class)

    return total_loss / len(dataloader), total_acc / len(dataloader)


def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, device,
          scheduler=None, checkpoint_path=None, patience=5):
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    epochs_no_improve = 0

    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_acc = test_step(model, val_dataloader, loss_fn, device)

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return results
